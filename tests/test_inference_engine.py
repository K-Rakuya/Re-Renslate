"""
tests/test_inference_engine.py
Comprehensive unit test suite for Renloc LLM Engine:
1. vLLM V1/V2 Adapter (vllm_adapter.py)
2. Hardened SGLang Adapter mitigating CVE-2026-7304 (sglang_adapter.py)
3. llama-cpp-python Hybrid Adapter (llamacpp_adapter.py)
4. GPU VRAM Monitor with _room_for_copy (vram_monitor.py)
5. Persistent Idempotent DAG Scheduler (dag_scheduler.py)
6. 3-Tier Semantic Router (router.py)
"""

import hashlib
import json
import os
import secrets
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import apsw
import pytest

from renloc_core.llm_engine.adapters.vllm_adapter import (
    BatchUpdate,
    MoveDirectionality,
    VllmCompatAdapter,
    VllmInterfaceType,
    VllmV1BatchUpdateProcessor,
    VllmV2ModelRunnerProcessor,
)
from renloc_core.llm_engine.adapters.sglang_adapter import (
    SecurityVulnerabilityError,
    SglangConstraintProcessor,
    SglangProcessorRegistry,
    SglangSecurityProxy,
    SignatureVerificationError,
    UnauthenticatedPayloadError,
    UnauthorizedAccessError,
    UntrustedProcessorClassError,
)
from renloc_core.llm_engine.adapters.llamacpp_adapter import (
    GBNFGrammarCompiler,
    GenerationResult,
    LlamaCppAdapter,
    LlamaCppLogitsProcessorCallback,
)
from renloc_core.llm_engine.vram_monitor import (
    ModelVramConfig,
    VramInfo,
    VramMonitor,
)
from renloc_core.llm_engine.dag_scheduler import (
    DagScheduler,
    DagTask,
    TaskStatus,
    compute_dag_task_key,
)
from renloc_core.llm_engine.router import (
    DialogueComplexityClassifier,
    DialogueMetadata,
    RouteResult,
    RoutingTier,
    SemanticDialogueRouter,
)


# ==============================================================================
# 1. vLLM Adapter Tests
# ==============================================================================

class DummySamplingParams:
    def __init__(
        self,
        allowed_token_ids=None,
        banned_token_ids=None,
        thinking_budget=None,
    ):
        self.allowed_token_ids = allowed_token_ids
        self.banned_token_ids = banned_token_ids
        self.thinking_budget = thinking_budget


class DummyContext:
    def __init__(self, expanded_idx_mapping=None, thinking_budget_exceeded=False):
        self.expanded_idx_mapping = expanded_idx_mapping
        self.thinking_budget_exceeded = thinking_budget_exceeded


class TestVllmAdapter:
    def test_import_light_param_validation(self):
        """Frontend can validate sampling parameters without GPU dependencies."""
        valid_params = DummySamplingParams(allowed_token_ids=[1, 2, 3])
        VllmV2ModelRunnerProcessor.validate_params(valid_params)
        VllmV1BatchUpdateProcessor.validate_params(valid_params)

        with pytest.raises(ValueError):
            invalid_params = DummySamplingParams(allowed_token_ids="not-a-collection")
            VllmV2ModelRunnerProcessor.validate_params(invalid_params)

    def test_v2_model_runner_slot_recycling_and_staged_writes(self):
        """Model Runner V2 reinitializes slots cleanly on add_request to avoid leakage."""
        proc = VllmV2ModelRunnerProcessor()
        p1 = DummySamplingParams(allowed_token_ids={10, 20}, thinking_budget=5)
        proc.add_request(0, p1)
        proc.apply_staged_writes()

        assert 0 in proc.req_states
        assert proc.req_states[0].allowed_token_ids == {10, 20}
        assert proc.req_states[0].thinking_budget == 5

        # Slot 0 recycled with new params
        p2 = DummySamplingParams(allowed_token_ids={30}, thinking_budget=10)
        proc.add_request(0, p2)
        proc.apply_staged_writes()

        assert proc.req_states[0].allowed_token_ids == {30}
        assert proc.req_states[0].step_count == 0

    def test_v2_speculative_decoding_expanded_idx_mapping(self):
        """Speculative decoding rows correctly map to request indices."""
        proc = VllmV2ModelRunnerProcessor()
        proc.add_request(0, DummySamplingParams(banned_token_ids={2}))
        proc.add_request(1, DummySamplingParams(banned_token_ids={3}))
        proc.apply_staged_writes()

        # 3 rows of logits: row 0 -> req 0, row 1 -> req 0 (speculative), row 2 -> req 1
        logits = [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ]
        ctx = DummyContext(expanded_idx_mapping=[0, 0, 1])
        masked = proc.apply(logits, ctx)

        assert masked[0][2] == float("-inf")
        assert masked[1][2] == float("-inf")
        assert masked[2][3] == float("-inf")

    def test_v1_batch_update_order_removed_added_moved(self):
        """vLLM V1 BatchUpdate processes removed -> added -> moved (including SWAP)."""
        proc = VllmV1BatchUpdateProcessor()
        # Initialize slot 0 and slot 1
        proc.slots[0] = DummySamplingParams(allowed_token_ids={10})
        proc.slots[1] = DummySamplingParams(allowed_token_ids={20})

        # Update: remove slot 0, add slot 2, swap slot 1 to slot 3
        update = BatchUpdate(
            removed=[0],
            added=[(2, DummySamplingParams(allowed_token_ids={30}))],
            moved=[(1, 3, MoveDirectionality.SWAP)],
        )
        proc.update_state(update)

        assert 0 not in proc.slots
        assert 1 not in proc.slots
        assert 2 in proc.slots
        assert 3 in proc.slots

    def test_inf_logit_safe_recovery(self):
        """Safe recovery restores argmax when all logits are masked to -inf."""
        proc = VllmV1BatchUpdateProcessor()
        p = DummySamplingParams(allowed_token_ids={999})  # Token out of vocab
        proc.slots[0] = p

        logits = [[1.0, 5.0, 2.0]]
        masked = proc.apply(logits)

        # Token 1 had highest initial logit (5.0), so it should be restored
        assert masked[0][1] == 5.0
        assert masked[0][0] == float("-inf")

    def test_vllm_compat_adapter_detection(self):
        """VllmCompatAdapter gracefully detects and instantiates appropriate processor."""
        detected = VllmCompatAdapter.detect_interface()
        assert detected in (
            VllmInterfaceType.V2_MODEL_RUNNER,
            VllmInterfaceType.V1_BATCH_UPDATE,
            VllmInterfaceType.COMPAT_FALLBACK,
        )
        instance = VllmCompatAdapter.create_processor()
        assert instance is not None


# ==============================================================================
# 2. Hardened SGLang Adapter & CVE-2026-7304 Mitigation Tests
# ==============================================================================

class TestSglangSecurityAdapter:
    def test_strict_localhost_binding(self):
        """SGLang proxy strictly prohibits binding to 0.0.0.0 or external interfaces."""
        with pytest.raises(ValueError, match="127.0.0.1"):
            SglangSecurityProxy(host="0.0.0.0")

        with pytest.raises(ValueError, match="127.0.0.1"):
            SglangSecurityProxy(host="192.168.1.100")

        proxy = SglangSecurityProxy(host="127.0.0.1")
        assert proxy.host == "127.0.0.1"

    def test_internal_bearer_token_authentication(self):
        """Internal bearer token is validated in constant time."""
        proxy = SglangSecurityProxy()
        valid_header = f"Bearer {proxy.auth_token}"
        proxy.authenticate_request(valid_header)  # No exception

        with pytest.raises(UnauthorizedAccessError):
            proxy.authenticate_request("Bearer invalid-token")

        with pytest.raises(UnauthorizedAccessError):
            proxy.authenticate_request("")

    def test_cve_2026_7304_rejection_of_raw_unsigned_dill_payload(self):
        """CRITICAL: Rejects 100% of raw unauthenticated dill / pickle payloads."""
        proxy = SglangSecurityProxy()

        # Simulate attacker sending raw pickled hex payload without HMAC signature
        malicious_hex = "80049512000000000000008c086f732e73797374656d"
        raw_payload = f"RAW_UNAUTHENTICATED:{malicious_hex}"

        with pytest.raises(UnauthenticatedPayloadError, match="CVE-2026-7304"):
            proxy.verify_and_extract_payload(raw_payload)

        # Also test bare hex string
        with pytest.raises(UnauthenticatedPayloadError):
            proxy.verify_and_extract_payload(malicious_hex)

    def test_cve_2026_7304_tampered_hmac_signature_rejection(self):
        """Rejects payload if HMAC signature or payload content was altered."""
        proxy = SglangSecurityProxy()
        original_bytes = b'{"processor": "SglangConstraintProcessor"}'
        signed_str = proxy.sign_payload(original_bytes)

        # Tamper with the hex signature
        parts = signed_str.split(":")
        tampered_sig = "00" * 32
        tampered_str = f"HMAC_SHA256:{tampered_sig}:{parts[2]}"

        with pytest.raises(SignatureVerificationError):
            proxy.verify_and_extract_payload(tampered_str)

        # Tamper with the payload data
        tampered_data = parts[2] + "ff"
        tampered_payload = f"HMAC_SHA256:{parts[1]}:{tampered_data}"
        with pytest.raises(SignatureVerificationError):
            proxy.verify_and_extract_payload(tampered_payload)

    def test_valid_hmac_signing_and_processor_instantiation(self):
        """Signed processor serializes and deserializes safely."""
        proxy = SglangSecurityProxy()
        processor = SglangConstraintProcessor(allowed_token_ids={1, 2, 3})

        signed_str = processor.to_str(security_proxy=proxy)
        assert signed_str.startswith("HMAC_SHA256:")

        recovered = SglangConstraintProcessor.from_str(signed_str, security_proxy=proxy)
        assert recovered.allowed_token_ids == {1, 2, 3}

    def test_processor_registry_whitelisting(self):
        """Registry disallows unregistered or unauthorized processor classes."""
        with pytest.raises(UntrustedProcessorClassError):
            SglangProcessorRegistry.get("MaliciousExecutionProcessor")

        cls = SglangProcessorRegistry.get("RenlocTagConstraintProcessor")
        assert cls == SglangConstraintProcessor


# ==============================================================================
# 3. llama-cpp-python Hybrid Adapter Tests
# ==============================================================================

class TestLlamaCppAdapter:
    def test_gbnf_grammar_compilation_under_threshold(self):
        """When N <= 5 tags, GBNF grammar compiler generates valid grammar rules."""
        tags = ["{b}", "{/b}", "[player_name]"]
        gbnf = GBNFGrammarCompiler.compile_grammar_for_tags(tags)

        assert "root ::=" in gbnf
        assert 'tag0 ::= "{b}"' in gbnf
        assert 'tag1 ::= "{/b}"' in gbnf
        assert 'tag2 ::= "[player_name]"' in gbnf
        assert "tag ::= tag0 | tag1 | tag2" in gbnf

    def test_logits_processor_callback_over_threshold(self):
        """When N > 5 tags, stateful callback handles constraints without grammar explosion."""
        tags = ["{b}", "{/b}", "{i}", "{/i}", "{color=#fff}", "{/color}"]
        assert len(tags) > 5

        callback = LlamaCppLogitsProcessorCallback(tags=tags)
        logits = [1.0, 2.0, 3.0]
        out_logits = callback([101], logits)
        assert out_logits == logits

    def test_llamacpp_adapter_mode_selection(self):
        """LlamaCppAdapter selects GBNF mode for N <= 5 and callback mode for N > 5."""
        adapter = LlamaCppAdapter()

        # N <= 5: GBNF mode
        res1 = adapter.generate("Hello", tags=["{b}", "{/b}"])
        assert res1.enforcement_mode == "gbnf"

        # N > 5: Logits callback mode
        res2 = adapter.generate("Hello", tags=["{1}", "{2}", "{3}", "{4}", "{5}", "{6}"])
        assert res2.enforcement_mode == "logits_callback"

        # N == 0: Unconstrained
        res3 = adapter.generate("Hello", tags=[])
        assert res3.enforcement_mode == "none"


# ==============================================================================
# 4. GPU VRAM Monitor Tests (_room_for_copy)
# ==============================================================================

class TestVramMonitor:
    def test_vram_info_query(self):
        """Queries GPU or simulated memory successfully."""
        monitor = VramMonitor()
        info = monitor.get_vram_info()
        assert info.total_bytes > 0
        assert info.free_bytes >= 0
        assert info.total_mb > 0

    def test_room_for_copy_calculation(self):
        """Verifies _room_for_copy reserves at least min_room_for_copy_bytes or percentage."""
        monitor = VramMonitor(min_room_for_copy_bytes=512 * 1024 * 1024, room_for_copy_ratio=0.20)
        # Case 1: 1 GB free -> 20% is 204.8 MB -> clamped to min 512 MB
        free_1gb = 1024 * 1024 * 1024
        assert monitor.calculate_room_for_copy(free_1gb) == 512 * 1024 * 1024

        # Case 2: 4 GB free -> 20% is 819.2 MB -> exceeds min 512 MB
        free_4gb = 4 * 1024 * 1024 * 1024
        assert monitor.calculate_room_for_copy(free_4gb) == int(free_4gb * 0.20)

    def test_dynamic_batch_contraction_on_oom_pressure(self):
        """Batch size automatically contracts when free VRAM drops."""
        # Simulated tight VRAM: only 600 MB free
        tight_monitor = VramMonitor(
            mock_mode=True,
            mock_free_bytes=600 * 1024 * 1024,
            min_room_for_copy_bytes=512 * 1024 * 1024,
        )
        batch = tight_monitor.calculate_adaptive_batch_size(target_batch_size=32)
        assert batch == 1  # Contracted down to 1

        # Simulated ample VRAM: 6 GB free
        ample_monitor = VramMonitor(
            mock_mode=True,
            mock_free_bytes=6 * 1024 * 1024 * 1024,
            min_room_for_copy_bytes=512 * 1024 * 1024,
        )
        batch_ample = ample_monitor.calculate_adaptive_batch_size(target_batch_size=16)
        assert batch_ample == 16  # Preserves full requested batch

    def test_oom_risk_detection(self):
        """Accurately identifies dangerous execution sizes exceeding safe effective memory."""
        monitor = VramMonitor(
            mock_mode=True,
            mock_free_bytes=800 * 1024 * 1024,
            min_room_for_copy_bytes=512 * 1024 * 1024,
        )
        # Effective VRAM is 800 - 512 = 288 MB.
        # Batch of 100 requests will exceed 288 MB.
        assert monitor.check_oom_risk(batch_size=100) is True
        assert monitor.check_oom_risk(batch_size=1) is False


# ==============================================================================
# 5. Persistent Idempotent DAG Scheduler Tests
# ==============================================================================

class TestDagScheduler:
    @pytest.fixture
    def scheduler(self, tmp_path: Path):
        db_file = tmp_path / "test_dag.db"
        sched = DagScheduler(str(db_file))
        yield sched
        sched.close()

    def test_task_key_determinism(self):
        """sha256(source_hash | icfg_context_id | glossary_version | model_id) is deterministic."""
        k1 = compute_dag_task_key("hashA", "ctx1", "v1", "modelX")
        k2 = compute_dag_task_key("hashA", "ctx1", "v1", "modelX")
        k3 = compute_dag_task_key("hashB", "ctx1", "v1", "modelX")
        assert k1 == k2
        assert k1 != k3
        assert len(k1) == 64

    def test_idempotent_task_submission(self, scheduler: DagScheduler):
        """Re-submitting the same task returns existing state without resetting progress."""
        task1 = scheduler.submit_translation_task("Hello", "h1", "c1", "v1", "m1")
        assert task1.status == TaskStatus.PENDING

        # Lease and progress to DRAFTED
        leased = scheduler.lease_tasks("worker-1", limit=1)
        assert len(leased) == 1
        scheduler.transition_state(task1.task_id, TaskStatus.DRAFTED, {"draft": "Xin chào"})

        # Re-submit same task key: MUST return DRAFTED without resetting
        task2 = scheduler.submit_translation_task("Hello", "h1", "c1", "v1", "m1")
        assert task2.status == TaskStatus.DRAFTED
        assert task2.result["draft"] == "Xin chào"

    def test_monotonic_milestone_progression(self, scheduler: DagScheduler):
        """Full pipeline: PENDING -> LEASED -> DRAFTED -> REFINED -> QE_SCORED -> COMMITTED."""
        task = scheduler.submit_translation_task("Dialogue line", "h2", "c2", "v1", "m1")
        assert task.status == TaskStatus.PENDING

        # LEASED
        scheduler.lease_tasks("worker-alpha", limit=1)
        t = scheduler.get_task(task.task_id)
        assert t.status == TaskStatus.LEASED

        # DRAFTED
        scheduler.transition_state(task.task_id, TaskStatus.DRAFTED, {"draft": "Bản dịch thô"})
        assert scheduler.get_task(task.task_id).status == TaskStatus.DRAFTED

        # REFINED
        scheduler.transition_state(task.task_id, TaskStatus.REFINED, {"refined": "Bản dịch tinh chỉnh"})
        assert scheduler.get_task(task.task_id).status == TaskStatus.REFINED

        # QE_SCORED
        scheduler.transition_state(task.task_id, TaskStatus.QE_SCORED, {"qe_score": 0.94})
        assert scheduler.get_task(task.task_id).status == TaskStatus.QE_SCORED

        # COMMITTED
        scheduler.transition_state(task.task_id, TaskStatus.COMMITTED, {"target_text": "Bản dịch hoàn thiện"})
        final = scheduler.get_task(task.task_id)
        assert final.status == TaskStatus.COMMITTED
        assert final.result["target_text"] == "Bản dịch hoàn thiện"

    def test_crash_recovery_resets_expired_leases(self, scheduler: DagScheduler):
        """Crash simulation: interrupted worker leases recover to PENDING."""
        task = scheduler.submit_translation_task("Crash test line", "h3", "c3", "v1", "m1")
        # Lease with 0-second duration (immediately expired)
        scheduler.lease_tasks("worker-crash", limit=1, lease_duration_sec=0)
        assert scheduler.get_task(task.task_id).status == TaskStatus.LEASED

        # Simulate restart recovery
        recovered = scheduler.recover_after_crash()
        assert recovered == 1
        assert scheduler.get_task(task.task_id).status == TaskStatus.PENDING


# ==============================================================================
# 6. 3-Tier Semantic Router Tests
# ==============================================================================

class TestSemanticRouter:
    @pytest.fixture
    def router_with_db(self, tmp_path: Path):
        db_path = str(tmp_path / "router_test.db")
        conn = apsw.Connection(db_path)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS glossary (
            src_term TEXT PRIMARY KEY,
            tgt_term TEXT
        );
        CREATE TABLE IF NOT EXISTS translation_memory (
            source_hash TEXT PRIMARY KEY,
            target_text TEXT,
            similarity_score REAL
        );
        """)
        # Populate fixture data
        cur.execute("INSERT INTO glossary VALUES ('Eileen', 'Eileen-san');")
        cur.execute(
            "INSERT INTO translation_memory VALUES (?, 'Lâu rồi không gặp!', 1.0);",
            (hashlib.sha256("Long time no see!".encode()).hexdigest(),),
        )

        router = SemanticDialogueRouter(db_conn_or_path=conn)
        yield router
        conn.close()

    def test_tier_1_pure_tags_and_numbers(self, router_with_db: SemanticDialogueRouter):
        """Tier 1: Formatting pauses and numbers resolve instantly with 0 tokens."""
        res1 = router_with_db.route_and_execute("{w=1.0}")
        assert res1.tier == RoutingTier.TIER_1_DETERMINISTIC
        assert res1.target_text == "{w=1.0}"
        assert res1.score == 1.0

        res2 = router_with_db.route_and_execute("100% {nw}")
        assert res2.tier == RoutingTier.TIER_1_DETERMINISTIC

    def test_tier_1_glossary_and_tm_lookup(self, router_with_db: SemanticDialogueRouter):
        """Tier 1: Exact matches in TM or Glossary bypass LLM."""
        res_glossary = router_with_db.route_and_execute("Eileen")
        assert res_glossary.tier == RoutingTier.TIER_1_DETERMINISTIC
        assert res_glossary.target_text == "Eileen-san"

        res_tm = router_with_db.route_and_execute("Long time no see!")
        assert res_tm.tier == RoutingTier.TIER_1_DETERMINISTIC
        assert res_tm.target_text == "Lâu rồi không gặp!"

    def test_tier_2_standard_dialogue(self, router_with_db: SemanticDialogueRouter):
        """Tier 2: Standard single-pass dialogue routes to lightweight local model."""
        res = router_with_db.route_and_execute(
            "Let us go to the market today.",
            DialogueMetadata(speaker="Alice"),
        )
        assert res.tier == RoutingTier.TIER_2_LIGHT_ONE_PASS
        assert res.method == "light_one_pass"

    def test_tier_3_high_complexity_dialogue(self, router_with_db: SemanticDialogueRouter):
        """Tier 3: Complex scenes trigger TEaR + MBR (K candidates) + isolated QE scoring."""
        meta = DialogueMetadata(
            speaker="???",
            is_dynamic_speaker=True,
            has_pronoun_shift=True,
            tags=["{i}", "{/i}"],
        )
        res = router_with_db.route_and_execute(
            "You cannot escape your true fate, {i}Hero{/i}!",
            meta,
        )
        assert res.tier == RoutingTier.TIER_3_DEEP_TEAR_MBR
        assert res.method == "tear_mbr_qe"
        assert len(res.candidates) >= 4
        assert res.score is not None
        assert res.score >= 0.80
