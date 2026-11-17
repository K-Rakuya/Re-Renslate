//! job_guard.rs
//! Windows Job Object and POSIX Process Group / Death-pipe guards to prevent orphan sidecar processes.

use std::process::Child;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;
use tracing::info;

#[derive(Debug, Error)]
pub enum GuardError {
    #[error("System error: {0}")]
    System(String),
}

pub struct ProcessGuard {
    child_pid: u32,
    cleaned: Arc<AtomicBool>,
    #[cfg(unix)]
    death_pipe_write: Option<std::os::unix::io::OwnedFd>,
}

impl ProcessGuard {
    #[cfg(windows)]
    pub fn new(child: &Child) -> Result<Self, GuardError> {
        use windows_sys::Win32::Foundation::*;
        use windows_sys::Win32::System::JobObjects::*;
        use windows_sys::Win32::System::Threading::*;

        let child_pid = child.id();
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                return Err(GuardError::System("Failed to create Job Object".into()));
            }

            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

            let set_res = SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const _,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            if set_res == 0 {
                CloseHandle(job);
                return Err(GuardError::System("Failed to set Job Object limits".into()));
            }

            let process_handle = OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, 0, child_pid);
            if process_handle.is_null() {
                CloseHandle(job);
                return Err(GuardError::System(
                    "Failed to open child process handle".into(),
                ));
            }

            let assign_res = AssignProcessToJobObject(job, process_handle);
            CloseHandle(process_handle);

            if assign_res == 0 {
                CloseHandle(job);
                return Err(GuardError::System(
                    "Failed to assign process to Job Object".into(),
                ));
            }
        }

        info!("Attached child PID {} to Windows Job Object", child_pid);
        Ok(Self {
            child_pid,
            cleaned: Arc::new(AtomicBool::new(false)),
        })
    }

    #[cfg(unix)]
    pub fn new(
        child: &Child,
        death_pipe_write: Option<std::os::unix::io::OwnedFd>,
    ) -> Result<Self, GuardError> {
        let child_pid = child.id();
        info!("Registered POSIX process guard for child PID {}", child_pid);
        Ok(Self {
            child_pid,
            cleaned: Arc::new(AtomicBool::new(false)),
            death_pipe_write,
        })
    }

    #[cfg(unix)]
    pub fn spawn_guarded(cmd: &mut std::process::Command) -> Result<(Child, Self), GuardError> {
        use std::os::unix::io::FromRawFd;
        use std::os::unix::process::CommandExt;

        // Create pipe for death-pipe detection
        let mut pipe_fds = [0i32; 2];
        let res = unsafe { libc::pipe(pipe_fds.as_mut_ptr()) };
        if res != 0 {
            return Err(GuardError::System("Failed to create death-pipe".into()));
        }

        let read_fd = pipe_fds[0];
        let write_fd = pipe_fds[1];

        // Ensure child runs in its own process group (setpgid(0, 0))
        cmd.process_group(0);

        // Keep read_fd open across exec so the child can read from it
        unsafe {
            libc::fcntl(read_fd, libc::F_SETFD, 0);
        }
        cmd.env("RENSLATE_DEATH_PIPE_FD", read_fd.to_string());

        // On Linux, use prctl(PR_SET_PDEATHSIG, SIGKILL) in pre_exec so kernel auto-kills
        // child if parent crashes or is killed with SIGKILL
        #[cfg(target_os = "linux")]
        unsafe {
            cmd.pre_exec(move || {
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL);
                Ok(())
            });
        }

        let child = cmd.spawn().map_err(|e| {
            unsafe {
                libc::close(read_fd);
                libc::close(write_fd);
            }
            GuardError::System(e.to_string())
        })?;

        // In parent, close read end, keep write end (OwnedFd automatically closes on drop)
        unsafe {
            libc::close(read_fd);
        }
        let death_pipe_write = unsafe { std::os::unix::io::OwnedFd::from_raw_fd(write_fd) };

        let guard = Self::new(&child, Some(death_pipe_write))?;
        Ok((child, guard))
    }

    #[cfg(windows)]
    pub fn spawn_guarded(cmd: &mut std::process::Command) -> Result<(Child, Self), GuardError> {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW = 0x08000000
        cmd.creation_flags(0x08000000);

        let child = cmd.spawn().map_err(|e| GuardError::System(e.to_string()))?;
        let guard = Self::new(&child)?;
        Ok((child, guard))
    }

    pub fn terminate(&mut self) {
        if self.cleaned.swap(true, Ordering::SeqCst) {
            return;
        }

        #[cfg(unix)]
        {
            use nix::sys::signal::{kill, Signal};
            use nix::unistd::Pid;

            // Close death-pipe to signal EOF to child
            self.death_pipe_write.take();

            let pid = Pid::from_raw(self.child_pid as i32);
            // Send SIGTERM to process group (-pgid)
            let pgid = -pid.as_raw();
            let _ = kill(Pid::from_raw(pgid), Signal::SIGTERM);

            // Wait with timeout up to 3000ms as specified in architecture design
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(3000);
            let mut still_alive = true;
            while std::time::Instant::now() < deadline {
                if kill(pid, None).is_err() {
                    still_alive = false;
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }

            if still_alive {
                // Send SIGKILL to the entire process group if still running
                let _ = kill(Pid::from_raw(pgid), Signal::SIGKILL);
                info!("POSIX process guard sent SIGKILL to group {}", pgid);
            } else {
                info!(
                    "POSIX process group {} exited gracefully after SIGTERM",
                    pgid
                );
            }
        }

        #[cfg(windows)]
        {
            info!("Windows Job Object automatically terminates child on handle close.");
        }
    }
}

impl Drop for ProcessGuard {
    fn drop(&mut self) {
        self.terminate();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    #[cfg(unix)]
    fn test_process_guard_terminates_child_on_drop() {
        let mut cmd = Command::new("sleep");
        cmd.arg("60");

        let (mut child, guard) =
            ProcessGuard::spawn_guarded(&mut cmd).expect("Failed to spawn guarded sleep");
        let _pid = child.id();

        // Check child is running
        assert_eq!(child.try_wait().unwrap(), None);

        // Drop guard, triggering terminate()
        drop(guard);

        // Child should have been killed by SIGTERM / SIGKILL
        let status = child.wait().expect("Failed to wait on killed child");
        assert!(!status.success());
    }

    #[test]
    #[cfg(unix)]
    fn test_death_pipe_environment_set() {
        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg("echo $RENSLATE_DEATH_PIPE_FD");

        let (mut child, guard) = ProcessGuard::spawn_guarded(&mut cmd).expect("Failed to spawn");
        let status = child.wait().unwrap();
        assert!(status.success());
        drop(guard);
    }
}
