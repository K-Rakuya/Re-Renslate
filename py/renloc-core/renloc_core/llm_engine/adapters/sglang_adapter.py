"""
sglang_adapter.py
Hardened SGLang Adapter mitigating CVE-2026-7304 (CERT/CC VU#777338).

Vulnerability Context (CVE-2026-7304):
In SGLang v0.5.11–v0.5.12+, CustomLogitProcessor.from_str() invokes dill.loads() directly
on raw hex-encoded byte payloads received from untrusted HTTP POST requests without
HMAC verification or class whitelisting, resulting in unauthenticated Remote Code Execution (RCE).

Defense in Depth Architecture:
1. Strict Localhost Binding:
   - SGLang server and proxy are locked strictly to 127.0.0.1. Rejects binding to 0.0.0.0 or external interfaces.
2. Internal CSPRNG Auth Token:
   - All HTTP requests to SGLang proxy must provide a 256-bit CSPRNG bearer token generated at startup.
3. 32-Byte HMAC Payload Signing:
   - All custom logit processor payloads must be signed with a fresh 32-byte HMAC-SHA256 secret.
   - 100% rejection of raw, unauthenticated dill.loads payloads.
4. Server-Side Processor Registry (Whitelist):
   - Supports safe string-based registration ('RenlocConstraintProcessor') so clients only pass
     safe configuration parameters (custom_params) rather than arbitrary serialized code.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
from typing import Any, Callable, Dict, Optional, Set, Type, Union

logger = logging.getLogger(__name__)


# ==============================================================================
# Security Exceptions
# ==============================================================================

class SecurityVulnerabilityError(Exception):
    """Base exception for security-related rejections."""
    pass


class UnauthenticatedPayloadError(SecurityVulnerabilityError):
    """Raised when an unauthenticated / unsigned payload is detected (CVE-2026-7304 mitigation)."""
    pass


class SignatureVerificationError(SecurityVulnerabilityError):
    """Raised when an HMAC signature check fails (tampered or forged payload)."""
    pass


class UnauthorizedAccessError(SecurityVulnerabilityError):
    """Raised when internal proxy authentication fails."""
    pass


class UntrustedProcessorClassError(SecurityVulnerabilityError):
    """Raised when an unwhitelisted processor class is requested."""
    pass


# ==============================================================================
# Base Shim for SGLang CustomLogitProcessor
# ==============================================================================

try:
    from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor as _BaseCustomLogitProcessor
except ImportError:
    class _BaseCustomLogitProcessor:
        """Fallback base class when sglang is not installed."""
        pass


# ==============================================================================
# SGLang Security Proxy & HMAC Signer
# ==============================================================================

class SglangSecurityProxy:
    """
    Security supervisor and reverse proxy for local SGLang engine.
    Locks SGLang to 127.0.0.1, validates internal bearer tokens, and
    enforces 32-byte HMAC-SHA256 payload verification to defeat CVE-2026-7304.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 30000,
        auth_token: Optional[str] = None,
        hmac_secret: Optional[bytes] = None,
    ) -> None:
        # Enforce strict localhost lockdown
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError(
                f"Security violation: SGLang proxy must bind strictly to 127.0.0.1. "
                f"Attempted to bind to '{host}'."
            )
        self.host = "127.0.0.1"
        self.port = port

        # Internal 256-bit CSPRNG Auth Token
        self.auth_token = auth_token or secrets.token_hex(32)

        # 32-byte HMAC Secret for payload signing
        self.hmac_secret = hmac_secret or secrets.token_bytes(32)
        if len(self.hmac_secret) < 32:
            raise ValueError("HMAC secret must be at least 32 bytes.")

        self._allowed_classes: Set[str] = {
            "SglangConstraintProcessor",
            "RenlocTagConstraintProcessor",
        }

    def verify_auth_token(self, provided_token: Optional[str]) -> bool:
        """Verify internal bearer token using constant-time comparison."""
        if not provided_token:
            return False
        return hmac.compare_digest(self.auth_token, provided_token)

    def authenticate_request(self, auth_header: Optional[str]) -> None:
        """
        Validate Authorization header.
        Expected format: 'Bearer <token>'
        """
        if not auth_header:
            raise UnauthorizedAccessError("Missing Authorization header in SGLang proxy request.")

        parts = auth_header.strip().split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise UnauthorizedAccessError("Malformed Authorization header. Expected 'Bearer <token>'.")

        token = parts[1]
        if not self.verify_auth_token(token):
            raise UnauthorizedAccessError("Invalid internal auth token. Access denied.")

    def sign_payload(self, raw_bytes: bytes) -> str:
        """
        Sign raw serialized bytes with HMAC-SHA256 (32 bytes).
        Returns formatted string: 'HMAC_SHA256:<hex_signature>:<hex_payload>'
        """
        sig = hmac.new(self.hmac_secret, raw_bytes, hashlib.sha256).digest()
        return f"HMAC_SHA256:{sig.hex()}:{raw_bytes.hex()}"

    def verify_and_extract_payload(self, payload_str: str) -> bytes:
        """
        Validate HMAC-SHA256 signature and return the safe raw payload bytes.
        CRITICAL: Completely rejects raw dill/pickle strings lacking valid signature!
        """
        if not payload_str:
            raise UnauthenticatedPayloadError("Empty payload received.")

        if not payload_str.startswith("HMAC_SHA256:"):
            logger.critical(
                "CVE-2026-7304 DEFENSE TRIGGERED: Rejected raw unsigned payload! "
                "Attempt to invoke dill.loads without HMAC signature."
            )
            raise UnauthenticatedPayloadError(
                "CVE-2026-7304 Defense: Raw unsigned dill payload rejected! "
                "Payload must be signed with 32-byte HMAC-SHA256."
            )

        parts = payload_str.split(":", 2)
        if len(parts) != 3:
            raise UnauthenticatedPayloadError("Malformed HMAC-signed payload structure.")

        _, hex_sig, hex_payload = parts

        try:
            expected_sig = bytes.fromhex(hex_sig)
            payload_bytes = bytes.fromhex(hex_payload)
        except ValueError as err:
            raise UnauthenticatedPayloadError(f"Malformed hex encoding: {err}")

        # Constant-time comparison
        actual_sig = hmac.new(self.hmac_secret, payload_bytes, hashlib.sha256).digest()
        if not hmac.compare_digest(expected_sig, actual_sig):
            logger.critical("CVE-2026-7304 DEFENSE TRIGGERED: Tampered or forged HMAC signature!")
            raise SignatureVerificationError("HMAC signature verification failed. Untrusted payload discarded.")

        return payload_bytes

    def register_allowed_class(self, class_name: str) -> None:
        """Register an allowed processor class name."""
        self._allowed_classes.add(class_name)

    def is_class_allowed(self, class_name: str) -> bool:
        """Check if class is in whitelist."""
        return class_name in self._allowed_classes


# ==============================================================================
# SGLang Processor Registry (Whitelist Pattern)
# ==============================================================================

class SglangProcessorRegistry:
    """
    Server-side registry for allowed custom logit processors.
    Eliminates need for transmitting code over HTTP entirely.
    """
    _registry: Dict[str, Type[Any]] = {}

    @classmethod
    def register(cls, name: str, processor_cls: Type[Any]) -> None:
        cls._registry[name] = processor_cls
        logger.info(f"Registered safe SGLang processor: {name}")

    @classmethod
    def get(cls, name: str) -> Type[Any]:
        if name not in cls._registry:
            raise UntrustedProcessorClassError(
                f"Processor class '{name}' is not registered in safe SGLang whitelist."
            )
        return cls._registry[name]

    @classmethod
    def instantiate(cls, name: str, custom_params: Dict[str, Any]) -> Any:
        processor_cls = cls.get(name)
        return processor_cls(**custom_params)


# ==============================================================================
# SGLang Constraint Processor
# ==============================================================================

class SglangConstraintProcessor(_BaseCustomLogitProcessor):
    """
    CustomLogitProcessor for SGLang preserving Ren'Py interpolation and tags.
    Secured against CVE-2026-7304 via HMAC-SHA256 signed serialization.
    """

    def __init__(
        self,
        allowed_token_ids: Optional[Set[int]] = None,
        banned_token_ids: Optional[Set[int]] = None,
        tag_constraints: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.allowed_token_ids = allowed_token_ids
        self.banned_token_ids = banned_token_ids
        self.tag_constraints = tag_constraints or {}

    def to_str(self, security_proxy: Optional[SglangSecurityProxy] = None) -> str:
        """
        Serialize processor state.
        If security_proxy is supplied, payload is cryptographically signed with HMAC-SHA256.
        """
        # Safe JSON-serializable representation or serialized payload
        import json
        payload_data = {
            "processor": "SglangConstraintProcessor",
            "allowed_token_ids": list(self.allowed_token_ids) if self.allowed_token_ids else None,
            "banned_token_ids": list(self.banned_token_ids) if self.banned_token_ids else None,
            "tag_constraints": self.tag_constraints,
        }
        raw_bytes = json.dumps(payload_data, sort_keys=True).encode("utf-8")

        if security_proxy is not None:
            return security_proxy.sign_payload(raw_bytes)
        else:
            # Fallback warning: unsafely serialized payloads cannot be loaded by hardened proxy
            return f"RAW_UNAUTHENTICATED:{raw_bytes.hex()}"

    @classmethod
    def from_str(
        cls,
        payload_str: str,
        security_proxy: SglangSecurityProxy,
    ) -> SglangConstraintProcessor:
        """
        Secure deserialization requiring SglangSecurityProxy.
        Rejects raw unauthenticated payloads without 32-byte HMAC signature.
        """
        if not isinstance(security_proxy, SglangSecurityProxy):
            raise TypeError("A valid SglangSecurityProxy instance is required to safely deserialize processors.")

        # Authenticate and extract
        raw_bytes = security_proxy.verify_and_extract_payload(payload_str)

        import json
        data = json.loads(raw_bytes.decode("utf-8"))

        processor_name = data.get("processor")
        if not security_proxy.is_class_allowed(processor_name):
            raise UntrustedProcessorClassError(f"Class '{processor_name}' rejected by security proxy.")

        allowed_ids = set(data["allowed_token_ids"]) if data.get("allowed_token_ids") else None
        banned_ids = set(data["banned_token_ids"]) if data.get("banned_token_ids") else None
        tag_constraints = data.get("tag_constraints", {})

        return cls(
            allowed_token_ids=allowed_ids,
            banned_token_ids=banned_ids,
            tag_constraints=tag_constraints,
        )

    def __call__(self, logits: Any, custom_params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Apply logit masking during generation step.
        """
        params = custom_params or {}
        allowed = params.get("allowed_token_ids", self.allowed_token_ids)
        banned = params.get("banned_token_ids", self.banned_token_ids)

        has_torch = hasattr(logits, "is_cuda") or hasattr(logits, "masked_fill_")

        if has_torch:
            import torch
            if banned:
                banned_t = torch.tensor(list(banned), dtype=torch.long, device=logits.device)
                logits[..., banned_t] = float("-inf")
            if allowed:
                mask = torch.ones_like(logits, dtype=torch.bool)
                allowed_t = torch.tensor(list(allowed), dtype=torch.long, device=logits.device)
                mask[..., allowed_t] = False
                logits[mask] = float("-inf")
        else:
            # List / NumPy array fallback
            vocab_size = len(logits)
            if banned:
                for tid in banned:
                    if 0 <= tid < vocab_size:
                        logits[tid] = float("-inf")
            if allowed:
                allowed_set = set(allowed)
                for tid in range(vocab_size):
                    if tid not in allowed_set:
                        logits[tid] = float("-inf")

        return logits


# Register default safe processor
SglangProcessorRegistry.register("SglangConstraintProcessor", SglangConstraintProcessor)
SglangProcessorRegistry.register("RenlocTagConstraintProcessor", SglangConstraintProcessor)
