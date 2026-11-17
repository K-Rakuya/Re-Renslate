//! sidecar.rs
//! Renslate sidecar supervisor and lifecycle orchestrator.
//! Coordinates the renloc-bootstrapper, Job Object/process group guard,
//! and 256-bit CSPRNG authenticated RPC channel.

use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use thiserror::Error;
use tracing::info;

use crate::job_guard::{GuardError, ProcessGuard};
use crate::rpc_client::CsprngToken;

#[derive(Debug, Error)]
pub enum SidecarError {
    #[error("Guard failure: {0}")]
    Guard(#[from] GuardError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Bootstrapper binary not found at: {0}")]
    BinaryNotFound(PathBuf),
    #[error("Sidecar crashed or terminated prematurely")]
    ProcessTerminated,
}

pub struct SidecarSupervisor {
    bootstrapper_path: PathBuf,
    auth_token: CsprngToken,
    child: Option<Child>,
    guard: Option<ProcessGuard>,
}

impl SidecarSupervisor {
    pub fn new<P: AsRef<Path>>(bootstrapper_path: P) -> Self {
        Self {
            bootstrapper_path: bootstrapper_path.as_ref().to_path_buf(),
            auth_token: CsprngToken::generate(),
            child: None,
            guard: None,
        }
    }

    pub fn auth_token(&self) -> &CsprngToken {
        &self.auth_token
    }

    pub fn is_running(&mut self) -> bool {
        if let Some(child) = &mut self.child {
            matches!(child.try_wait(), Ok(None))
        } else {
            false
        }
    }

    pub fn start(&mut self, extra_args: &[&str]) -> Result<(), SidecarError> {
        if !self.bootstrapper_path.exists() {
            return Err(SidecarError::BinaryNotFound(self.bootstrapper_path.clone()));
        }

        let mut cmd = Command::new(&self.bootstrapper_path);
        cmd.args(extra_args);
        cmd.env("RENSLATE_AUTH_TOKEN", self.auth_token.as_hex());

        let (child, guard) = ProcessGuard::spawn_guarded(&mut cmd)?;
        info!(
            "Sidecar process spawned (PID: {}) with Job Object / POSIX guard",
            child.id()
        );

        self.child = Some(child);
        self.guard = Some(guard);
        Ok(())
    }

    pub fn stop(&mut self) {
        if let Some(mut guard) = self.guard.take() {
            guard.terminate();
        }
        if let Some(mut child) = self.child.take() {
            let _ = child.wait();
        }
        info!("Sidecar process terminated cleanly");
    }
}

impl Drop for SidecarSupervisor {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sidecar_supervisor_lifecycle() {
        // Use bash/sh or sleep for unit testing supervisor guard
        #[cfg(unix)]
        let test_bin = PathBuf::from("/bin/sh");
        #[cfg(windows)]
        let test_bin = PathBuf::from("cmd.exe");

        let mut supervisor = SidecarSupervisor::new(&test_bin);
        assert!(!supervisor.is_running());
        assert_eq!(supervisor.auth_token().as_bytes().len(), 32);

        #[cfg(unix)]
        let args = ["-c", "sleep 10"];
        #[cfg(windows)]
        let args = ["/C", "timeout /t 10"];

        let res = supervisor.start(&args);
        assert!(res.is_ok(), "Failed to start supervisor: {:?}", res.err());
        assert!(supervisor.is_running());

        supervisor.stop();
        assert!(!supervisor.is_running());
    }
}
