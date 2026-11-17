//! rpc_client.rs
//! MessagePack RPC client connecting Tauri desktop host to Python renloc-core sidecar.
//! Authenticated with a 256-bit CSPRNG session token passed during sidecar handshake.

use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
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
    #[error("RPC returned error: {0}")]
    Remote(String),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RpcEnvelope<T> {
    pub session_token: String,
    pub method: String,
    pub payload: T,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RpcResponse<T> {
    pub ok: bool,
    pub result: Option<T>,
    pub error: Option<String>,
}

pub struct RpcClient {
    pub session_token: String,
    pub endpoint: PathBuf,
}

impl RpcClient {
    pub fn new(endpoint: PathBuf) -> Self {
        // Generate cryptographically secure 256-bit session token
        let mut token_bytes = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut token_bytes);
        let session_token = hex::encode(token_bytes);

        Self {
            session_token,
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
            Ok(response.result.expect("Expected result on successful response"))
        } else {
            Err(RpcError::Remote(response.error.unwrap_or_else(|| "Unknown error".into())))
        }
    }
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
