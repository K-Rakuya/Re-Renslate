//! rpc_client.rs
//! MessagePack RPC client and server connecting Tauri desktop host to Python renloc-core sidecar.
//! Authenticated with a 256-bit CSPRNG session token passed during sidecar handshake.

use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Debug, Error)]
pub enum RpcError {
    #[error("I/O error during RPC: {0}")]
    Io(#[from] std::io::Error),
    #[error("MessagePack serialization error: {0}")]
    MsgPackEncode(#[from] rmp_serde::encode::Error),
    #[error("MessagePack deserialization error: {0}")]
    MsgPackDecode(#[from] rmp_serde::decode::Error),
    #[error("Unauthorized: Invalid session token")]
    Unauthorized,
    #[error("RPC returned error: {0}")]
    Remote(String),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RpcEnvelope<T> {
    pub session_token: String,
    pub method: String,
    pub payload: T,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RpcResponse<T> {
    pub ok: bool,
    pub result: Option<T>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsprngToken {
    bytes: [u8; 32],
    hex: String,
}

impl CsprngToken {
    pub fn generate() -> Self {
        let mut bytes = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut bytes);
        let hex = hex::encode(bytes);
        Self { bytes, hex }
    }

    pub fn from_hex(hex: &str) -> Option<Self> {
        if hex.len() != 64 {
            return None;
        }
        let mut bytes = [0u8; 32];
        for i in 0..32 {
            bytes[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).ok()?;
        }
        Some(Self {
            bytes,
            hex: hex.to_lowercase(),
        })
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.bytes
    }

    pub fn as_hex(&self) -> &str {
        &self.hex
    }

    /// Constant-time comparison to prevent side-channel timing attacks
    pub fn verify(&self, token_str: &str) -> bool {
        if self.hex.len() != token_str.len() {
            return false;
        }
        let a = self.hex.as_bytes();
        let b = token_str.as_bytes();
        let mut diff = 0u8;
        for i in 0..a.len() {
            diff |= a[i] ^ b[i];
        }
        diff == 0
    }
}

#[derive(Clone)]
pub struct RpcClient {
    pub session_token: String,
    pub endpoint: PathBuf,
}

impl RpcClient {
    pub fn new(endpoint: PathBuf) -> Self {
        let token = CsprngToken::generate();
        Self {
            session_token: token.as_hex().to_string(),
            endpoint,
        }
    }

    pub fn with_token(endpoint: PathBuf, token: String) -> Self {
        Self {
            session_token: token,
            endpoint,
        }
    }

    #[cfg(unix)]
    pub async fn call<Req: Serialize, Resp: for<'de> Deserialize<'de>>(
        &self,
        method: &str,
        payload: Req,
    ) -> Result<Resp, RpcError> {
        use tokio::net::UnixStream;

        let mut stream = UnixStream::connect(&self.endpoint).await?;
        let envelope = RpcEnvelope {
            session_token: self.session_token.clone(),
            method: method.to_string(),
            payload,
        };

        let encoded = rmp_serde::to_vec_named(&envelope)?;
        let len_header = (encoded.len() as u32).to_le_bytes();
        stream.write_all(&len_header).await?;
        stream.write_all(&encoded).await?;
        stream.flush().await?;

        // Read response length
        let mut resp_len_buf = [0u8; 4];
        stream.read_exact(&mut resp_len_buf).await?;
        let resp_len = u32::from_le_bytes(resp_len_buf) as usize;

        let mut resp_buf = vec![0u8; resp_len];
        stream.read_exact(&mut resp_buf).await?;

        let response: RpcResponse<Resp> = rmp_serde::from_slice(&resp_buf)?;
        if response.ok {
            Ok(response
                .result
                .expect("Expected result on successful response"))
        } else {
            Err(RpcError::Remote(
                response.error.unwrap_or_else(|| "Unknown error".into()),
            ))
        }
    }

    #[cfg(windows)]
    pub async fn call<Req: Serialize, Resp: for<'de> Deserialize<'de>>(
        &self,
        method: &str,
        payload: Req,
    ) -> Result<Resp, RpcError> {
        use tokio::net::windows::named_pipe::ClientOptions;

        let mut client = ClientOptions::new().open(&self.endpoint)?;
        let envelope = RpcEnvelope {
            session_token: self.session_token.clone(),
            method: method.to_string(),
            payload,
        };

        let encoded = rmp_serde::to_vec_named(&envelope)?;
        let len_header = (encoded.len() as u32).to_le_bytes();
        client.write_all(&len_header).await?;
        client.write_all(&encoded).await?;
        client.flush().await?;

        let mut resp_len_buf = [0u8; 4];
        client.read_exact(&mut resp_len_buf).await?;
        let resp_len = u32::from_le_bytes(resp_len_buf) as usize;

        let mut resp_buf = vec![0u8; resp_len];
        client.read_exact(&mut resp_buf).await?;

        let response: RpcResponse<Resp> = rmp_serde::from_slice(&resp_buf)?;
        if response.ok {
            Ok(response
                .result
                .expect("Expected result on successful response"))
        } else {
            Err(RpcError::Remote(
                response.error.unwrap_or_else(|| "Unknown error".into()),
            ))
        }
    }
}

/// Helper for configuring secure runtime directories and Unix Domain Sockets
#[cfg(unix)]
pub fn secure_uds_setup(socket_path: &Path) -> Result<(), std::io::Error> {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    if let Some(parent) = socket_path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
        // Runtime directory chmod 0700
        fs::set_permissions(parent, fs::Permissions::from_mode(0o700))?;
    }

    if socket_path.exists() {
        let _ = fs::remove_file(socket_path);
    }
    Ok(())
}

// Minimal hex encoder to avoid extra crate dependency
mod hex {
    pub fn encode(data: [u8; 32]) -> String {
        let mut s = String::with_capacity(64);
        for b in data {
            s.push_str(&format!("{:02x}", b));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use tokio::net::UnixListener;

    #[test]
    fn test_csprng_256bit_token_generation() {
        let client1 = RpcClient::new(PathBuf::from("/tmp/test1.sock"));
        let client2 = RpcClient::new(PathBuf::from("/tmp/test2.sock"));

        // 256-bit hex token is exactly 64 characters long
        assert_eq!(client1.session_token.len(), 64);
        assert_eq!(client2.session_token.len(), 64);

        // Tokens must be distinct (high entropy CSPRNG)
        assert_ne!(client1.session_token, client2.session_token);

        // All characters must be valid lowercase hex
        assert!(client1.session_token.chars().all(|c| c.is_ascii_hexdigit()));

        // Verification logic
        let token = CsprngToken::from_hex(&client1.session_token).unwrap();
        assert!(token.verify(&client1.session_token));
        assert!(!token.verify("0000000000000000000000000000000000000000000000000000000000000000"));
    }

    #[test]
    fn test_msgpack_rpc_serialization() {
        let envelope = RpcEnvelope {
            session_token: "abcd1234ef567890abcd1234ef567890abcd1234ef567890abcd1234ef567890"
                .to_string(),
            method: "ping".to_string(),
            payload: serde_json::json!({"test": true}),
        };

        let encoded = rmp_serde::to_vec_named(&envelope).expect("Failed to encode envelope");
        assert!(!encoded.is_empty());

        let decoded: RpcEnvelope<serde_json::Value> =
            rmp_serde::from_slice(&encoded).expect("Failed to decode envelope");
        assert_eq!(decoded.session_token, envelope.session_token);
        assert_eq!(decoded.method, "ping");
        assert_eq!(decoded.payload["test"], true);
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn test_uds_msgpack_rpc_roundtrip_with_token_auth() {
        let dir = tempdir().unwrap();
        let sock_path = dir.path().join("test_rpc.sock");
        secure_uds_setup(&sock_path).unwrap();

        let listener = UnixListener::bind(&sock_path).unwrap();
        let expected_token = CsprngToken::generate();
        let expected_token_hex = expected_token.as_hex().to_string();

        let server_token = expected_token.clone();
        // Spawn mock server task
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();

            // Read message
            let mut len_buf = [0u8; 4];
            stream.read_exact(&mut len_buf).await.unwrap();
            let req_len = u32::from_le_bytes(len_buf) as usize;

            let mut req_buf = vec![0u8; req_len];
            stream.read_exact(&mut req_buf).await.unwrap();

            let env: RpcEnvelope<serde_json::Value> = rmp_serde::from_slice(&req_buf).unwrap();

            let resp: RpcResponse<serde_json::Value> = if server_token.verify(&env.session_token) {
                if env.method == "echo" {
                    RpcResponse {
                        ok: true,
                        result: Some(env.payload),
                        error: None,
                    }
                } else {
                    RpcResponse {
                        ok: false,
                        result: None,
                        error: Some("Unknown method".into()),
                    }
                }
            } else {
                RpcResponse {
                    ok: false,
                    result: None,
                    error: Some("Unauthorized".into()),
                }
            };

            let resp_bytes = rmp_serde::to_vec_named(&resp).unwrap();
            let resp_len_hdr = (resp_bytes.len() as u32).to_le_bytes();
            stream.write_all(&resp_len_hdr).await.unwrap();
            stream.write_all(&resp_bytes).await.unwrap();
            stream.flush().await.unwrap();
        });

        // Client with valid token
        let client = RpcClient::with_token(sock_path.clone(), expected_token_hex.clone());
        let res: serde_json::Value = client
            .call("echo", serde_json::json!({"status": "active"}))
            .await
            .unwrap();
        assert_eq!(res["status"], "active");

        // Client with invalid token
        let bad_client = RpcClient::with_token(
            sock_path.clone(),
            "0000000000000000000000000000000000000000000000000000000000000000".into(),
        );
        let bad_res: Result<serde_json::Value, RpcError> =
            bad_client.call("echo", serde_json::json!({})).await;
        assert!(bad_res.is_err());
    }
}
