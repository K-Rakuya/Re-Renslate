//! job_guard.rs
//! Windows Job Object and POSIX Process Group / Death-pipe guards to prevent orphan sidecar processes.

use std::process::Child;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;
use tracing::{info, warn};

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
                return Err(GuardError::System("Failed to open child process handle".into()));
            }

            let assign_res = AssignProcessToJobObject(job, process_handle);
            CloseHandle(process_handle);

            if assign_res == 0 {
                CloseHandle(job);
                return Err(GuardError::System("Failed to assign process to Job Object".into()));
            }
        }

        info!("Attached child PID {} to Windows Job Object", child_pid);
        Ok(Self {
            child_pid,
            cleaned: Arc::new(AtomicBool::new(false)),
        })
    }

    #[cfg(unix)]
    pub fn new(child: &Child, death_pipe_write: Option<std::os::unix::io::OwnedFd>) -> Result<Self, GuardError> {
        let child_pid = child.id();
        info!("Registered POSIX process guard for child PID {}", child_pid);
        Ok(Self {
            child_pid,
            cleaned: Arc::new(AtomicBool::new(false)),
            death_pipe_write,
        })
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

            // Wait with timeout
            std::thread::sleep(std::time::Duration::from_millis(500));
            // Send SIGKILL if still running
            let _ = kill(Pid::from_raw(pgid), Signal::SIGKILL);
            info!("POSIX process guard terminated group {}", pgid);
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
