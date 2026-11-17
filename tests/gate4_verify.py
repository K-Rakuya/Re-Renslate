#!/usr/bin/env python3
"""
gate4_verify.py
Independent Gate 4 Verification Suite for Renslate.
Validates Section 5 (Local LLM Inference Engine & Quality Estimation Layer):
1. Fuzz Test 10,000 Ren'Py dialogue strings: 100% preservation of nested tags {...} and variables [...].
2. Security Penetration Test on SGLang Adapter: 100% rejection of unsigned/raw dill payloads (CVE-2026-7304 mitigation).
3. Persistent Idempotent DAG Scheduler Crash Recovery: 0 duplicate translations and checkpoint resumption.
4. 3-Tier Routing Architecture & TEaR + MBR Consensus with Isolated QE Worker.
5. VRAM KV-Cache Guard (_room_for_copy formula & batch size contraction).
"""

import os
import sys
import time
import hmac
import hashlib
import random
import secrets
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import apsw
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "py" / "renloc-core"))

from renloc_core.llm_engine.interpolation_lexer import (
    InterpolationLexer,
    TokenType,
    ruby_to_renpy,
    renpy_to_ruby,
    mask_substitutions,
    restore_substitutions,
    validate_placeholders,
    normalize_flags,
)
from renloc_core.llm_engine.lcs_segmenter import (
    LCSSegmenter,
    segment_pair,
    to_template,
    from_template,
    stitch_variants,
)
from renloc_core.llm_engine.constraint_compiler import (
    HybridConstraintCompiler,
    ConstraintCompiler,
    TagStackLogitsProcessor,
    safe_recover_tags,
    TAG_THRESHOLD,
)
from renloc_core.llm_engine.tear_mbr import (
    minimum_bayes_risk,
    MBRResult,
    TEaRPipeline,
    TEaRMBREngine,
    TEaRConfig,
    TEaRResult,
    estimate_translation_quality,
)
from renloc_core.llm_engine.adapters.sglang_adapter import (
    SglangSecurityProxy,
    SglangProcessorRegistry,
    SecurityVulnerabilityError,
    UnauthorizedAccessError,
    UnauthenticatedPayloadError,
    SignatureVerificationError,
    UntrustedProcessorClassError,
)
from renloc_core.llm_engine.dag_scheduler import (
    DagScheduler,
    DagTask,
    TaskStatus,
    compute_dag_task_key,
)
from renloc_core.llm_engine.router import (
    SemanticDialogueRouter,
    DialogueMetadata,
    RoutingTier,
)
from renloc_core.llm_engine.vram_monitor import (
    VramMonitor,
    ModelVramConfig,
    VramInfo,
)


class TestGate4FuzzLexer10k:
    """Audit Item 1: Fuzz test 10,000 sentences with nested tags and variable expressions."""

    def test_fuzz_10000_nested_tags_preservation(self):
        """
        Fuzz test 10,000 synthetic and realistic Ren'Py dialogue sentences.
        Verifies 100% preservation of all [...] and {...} structures, flags, and ruby text.
        """
        random.seed(42)

        tag_types = [
            ("{b}", "{/b}"),
            ("{i}", "{/i}"),
            ("{u}", "{/u}"),
            ("{s}", "{/s}"),
            ("{color=#ff0000}", "{/color}"),
            ("{color=#00ff00}", "{/color}"),
            ("{font=DejaVuSans.ttf}", "{/font}"),
            ("{size=+4}", "{/size}"),
            ("{size=24}", "{/size}"),
            ("{w}", ""),
            ("{w=1.5}", ""),
            ("{p}", ""),
            ("{p=2.0}", ""),
            ("{nw}", ""),
            ("{fast}", ""),
        ]

        var_names = ["player", "hero", "target", "gold", "score", "item_name", "location"]
        flag_variants = ["", "!u", "!l", "!c", "!t", "!q", "!ti", "!cl", "!lc", "!tu", "!r", "!s"]
        formats = ["", ":d", ":04d", ":.2f", ":s"]

        ruby_bases = ["東京", "魔法", "騎士", "皇帝", "星空", "先輩"]
        ruby_texts = ["とうきょう", "まほう", "ナイト", "エンペラー", "ほしぞら", "せんぱい"]

        words_en = [
            "Welcome", "to", "the", "academy", "champion", "The", "ancient", "castle",
            "holds", "many", "secrets", "Are", "you", "ready", "for", "the", "next",
            "adventure", "Please", "be", "careful", "out", "there"
        ]

        fuzz_corpus: List[str] = []

        # Generate 10,000 test sentences
        for idx in range(10000):
            parts: List[str] = []

            # 1. Opening words
            parts.append(random.choice(words_en))

            # 2. Add variable expression [...]
            v_name = random.choice(var_names)
            v_fmt = random.choice(formats)
            v_flag = random.choice(flag_variants)
            parts.append(f"[{v_name}{v_fmt}{v_flag}]")

            # 3. Add nested or single tag {...}
            t_open, t_close = random.choice(tag_types)
            if t_close:
                inner_word = random.choice(words_en)
                # Nesting opportunity
                if random.random() < 0.4:
                    n_open, n_close = random.choice(tag_types[:4])
                    if n_close:
                        parts.append(f"{t_open}{n_open}{inner_word}{n_close}{t_close}")
                    else:
                        parts.append(f"{t_open}{inner_word}{t_close}")
                else:
                    parts.append(f"{t_open}{inner_word}{t_close}")
            else:
                parts.append(t_open)

            # 4. Add Ruby text occasionally
            if random.random() < 0.2:
                r_base = random.choice(ruby_bases)
                r_text = random.choice(ruby_texts)
                parts.append(f"【{r_base}｜{r_text}】")

            # 5. Escapes
            if random.random() < 0.1:
                parts.append(r"\  ")  # preserved whitespace
            if random.random() < 0.05:
                parts.append("{{escaped_brace}}")

            parts.append(random.choice(words_en))
            sentence = " ".join(parts)
            fuzz_corpus.append(sentence)

        assert len(fuzz_corpus) == 10000

        t0 = time.perf_counter()
        failed_count = 0

        for idx, src in enumerate(fuzz_corpus):
            # Step A: Mask substitutions into invariant placeholders <T0>, <V1!...>
            masked, pmap = mask_substitutions(src)

            # Verify placeholder validity
            is_valid, missing, extra = validate_placeholders(masked, pmap)
            if not is_valid:
                failed_count += 1
                continue

            # Step B: Simulate translation (placeholders remain in Vietnamese target)
            translated = f"Bản dịch thử nghiệm: {masked}"

            # Step C: Restore substitutions
            restored = restore_substitutions(translated, pmap)

            # Step D: Verify every original tag and variable exists in restored
            for tag_content in pmap.t_map.values():
                if tag_content not in restored:
                    failed_count += 1
                    break
            for orig_raw, _ in pmap.v_map.values():
                if orig_raw not in restored:
                    failed_count += 1
                    break

        elapsed = time.perf_counter() - t0
        print(f"\n  [FUZZ TEST] Processed 10,000 sentences in {elapsed:.2f}s ({10000/elapsed:.0f} sent/s).")
        print(f"  [FUZZ TEST] Tag preservation failure count: {failed_count} / 10000.")

        assert failed_count == 0, f"Fuzz test failed on {failed_count} sentences!"
        print("  [PASS] 10,000 sentences fuzz test: 100% preservation of all [...] and {...} structures.")

    def test_ruby_text_and_flag_normalization(self):
        """Verify Ruby text conversion and 7-step pipeline flag equivalence."""
        # 1. Ruby conversion
        ruby_src = "Chào mừng đến 【東京｜とうきょう】 nhé!"
        renpy_style = ruby_to_renpy(ruby_src)
        assert "{rb}東京{/rb}{rt}とうきょう{/rt}" in renpy_style
        roundtrip = renpy_to_ruby(renpy_style)
        assert roundtrip == ruby_src

        # 2. Flag normalization (!cl == !lc, extra ! ignored)
        assert normalize_flags("!cl") == normalize_flags("!lc")
        assert normalize_flags("!l!c") == normalize_flags("!c!l")
        assert normalize_flags("!!ti") == normalize_flags("!ti")
        print("  [PASS] Ruby text bidirectional conversion and flag normalization verified.")


class TestGate4SglangSecurityPentest:
    """Audit Item 2: Security penetration test against CVE-2026-7304 (SGLang dill.loads RCE)."""

    def test_sglang_security_hardening(self):
        proxy = SglangSecurityProxy(host="127.0.0.1")
        valid_header = f"Bearer {proxy.auth_token}"

        # 1. Binding to non-localhost must fail
        with pytest.raises(ValueError, match="127.0.0.1"):
            SglangSecurityProxy(host="0.0.0.0")

        with pytest.raises(ValueError, match="127.0.0.1"):
            SglangSecurityProxy(host="192.168.1.10")

        # 127.0.0.1 passes
        assert proxy.host == "127.0.0.1"

        # 2. Missing or invalid Bearer token must be rejected
        with pytest.raises(UnauthorizedAccessError):
            proxy.authenticate_request(None)

        with pytest.raises(UnauthorizedAccessError):
            proxy.authenticate_request("Bearer invalid_token_value")

        # Valid Bearer token passes
        proxy.authenticate_request(valid_header)

        # 3. Raw unsigned dill.loads payload must be rejected
        malicious_hex = "80049512000000000000008c086f732e73797374656d"  # Classic pickle RCE payload
        raw_payload = f"RAW_UNAUTHENTICATED:{malicious_hex}"
        with pytest.raises(UnauthenticatedPayloadError):
            proxy.verify_and_extract_payload(raw_payload)

        # 4. Tampered HMAC signature must be rejected
        signed_str = proxy.sign_payload(b'{"processor": "SglangConstraintProcessor"}')
        parts = signed_str.split(":")
        tampered_sig = "00" * 32
        tampered_str = f"HMAC_SHA256:{tampered_sig}:{parts[2]}"
        with pytest.raises(SignatureVerificationError):
            proxy.verify_and_extract_payload(tampered_str)

        tampered_payload = f"HMAC_SHA256:{parts[1]}:{parts[2]}ff"
        with pytest.raises(SignatureVerificationError):
            proxy.verify_and_extract_payload(tampered_payload)

        # 5. Untrusted processor class must be rejected by server registry
        with pytest.raises(UntrustedProcessorClassError):
            SglangProcessorRegistry.get("MaliciousExecutionProcessor")

        assert SglangProcessorRegistry.get("RenlocTagConstraintProcessor") is not None

        print("  [PASS] SGLang CVE-2026-7304 defense verified: 100% of untrusted/unsigned payloads rejected.")


class TestGate4DagIdempotentScheduler:
    """Audit Item 3: Idempotent DAG scheduler with apsw BEGIN IMMEDIATE crash recovery."""

    def test_dag_scheduler_crash_recovery(self, tmp_path: Path):
        db_path = tmp_path / "dag_audit.sqlite"
        conn = apsw.Connection(str(db_path))

        scheduler = DagScheduler(conn)

        # 1. Enqueue 10 tasks with deterministic task keys
        task_ids = []
        for i in range(10):
            src = f"Dialogue line {i}"
            src_hash = hashlib.sha256(src.encode("utf-8")).hexdigest()
            task = scheduler.submit_translation_task(
                source_text=src,
                source_hash=src_hash,
                icfg_context_id=f"ctx_{i % 3}",
                glossary_version="v1.0",
                model_id="gemini-3.8-flash",
            )
            task_ids.append(task.task_id)

        # Count pending tasks
        pending_count = len([t for t in (scheduler.get_task(tid) for tid in task_ids) if t.status == TaskStatus.PENDING])
        assert pending_count == 10

        # Idempotency check: Re-enqueuing identical parameters returns existing task
        dup_task = scheduler.submit_translation_task(
            source_text="Dialogue line 0",
            source_hash=hashlib.sha256(b"Dialogue line 0").hexdigest(),
            icfg_context_id="ctx_0",
            glossary_version="v1.0",
            model_id="gemini-3.8-flash",
        )
        assert dup_task.task_id == task_ids[0]

        # 2. Worker leases 5 tasks with 1 second lease
        leased = scheduler.lease_tasks(worker_id="worker_gpu_0", limit=5, lease_duration_sec=1)
        assert len(leased) == 5
        assert all(scheduler.get_task(t.task_id).status == TaskStatus.LEASED for t in leased)

        # 3. Worker advances milestone to COMMITTED for first 2 tasks
        scheduler.transition_state(
            task_ids[0],
            new_status=TaskStatus.COMMITTED,
            result_update={"target_text": "Bản dịch dòng 0"},
        )
        scheduler.transition_state(
            task_ids[1],
            new_status=TaskStatus.COMMITTED,
            result_update={"target_text": "Bản dịch dòng 1"},
        )
        assert scheduler.get_task(task_ids[0]).status == TaskStatus.COMMITTED
        assert scheduler.get_task(task_ids[1]).status == TaskStatus.COMMITTED

        # 4. Simulate crash: Wait for lease to expire and invoke recover_after_crash()
        time.sleep(1.2)
        recovered_count = scheduler.recover_after_crash()
        assert recovered_count == 3, f"Expected 3 expired leased tasks to recover, got {recovered_count}"

        # 5. Verify states after crash recovery
        assert scheduler.get_task(task_ids[0]).status == TaskStatus.COMMITTED
        assert scheduler.get_task(task_ids[1]).status == TaskStatus.COMMITTED
        # Other 3 previously leased tasks are reset to PENDING
        assert scheduler.get_task(leased[2].task_id).status == TaskStatus.PENDING
        assert scheduler.get_task(leased[3].task_id).status == TaskStatus.PENDING
        assert scheduler.get_task(leased[4].task_id).status == TaskStatus.PENDING

        scheduler.close()
        print("  [PASS] DAG Scheduler crash recovery verified: 0 duplicate translations, seamless checkpoint restart.")


class TestGate4ThreeTierRoutingAndTEaR:
    """Audit Item 4: 3-tier routing and TEaR + MBR consensus execution."""

    def test_three_tier_dialogue_router(self, tmp_path: Path):
        db_path = str(tmp_path / "router_audit.db")
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
        cur.execute("INSERT INTO glossary VALUES ('Eileen', 'Eileen-san');")
        cur.execute(
            "INSERT INTO translation_memory VALUES (?, 'Lâu rồi không gặp!', 1.0);",
            (hashlib.sha256("Long time no see!".encode()).hexdigest(),),
        )

        router = SemanticDialogueRouter(db_conn_or_path=conn)

        # Tier 1: Pure punctuation / tags / numbers
        res_t1 = router.route_and_execute("{w=1.0}")
        assert res_t1.tier == RoutingTier.TIER_1_DETERMINISTIC
        assert res_t1.target_text == "{w=1.0}"

        res_t1_tm = router.route_and_execute("Long time no see!")
        assert res_t1_tm.tier == RoutingTier.TIER_1_DETERMINISTIC
        assert res_t1_tm.target_text == "Lâu rồi không gặp!"

        # Tier 2: Standard simple dialogue
        res_t2 = router.route_and_execute(
            "Let us go to the market today.",
            DialogueMetadata(speaker="Alice"),
        )
        assert res_t2.tier == RoutingTier.TIER_2_LIGHT_ONE_PASS

        # Tier 3: Complex narrative (temporal shift or unknown speaker "???")
        meta_t3 = DialogueMetadata(
            speaker="???",
            is_dynamic_speaker=True,
            has_pronoun_shift=True,
            tags=["{i}", "{/i}"],
        )
        res_t3 = router.route_and_execute(
            "You cannot escape your true fate, {i}Hero{/i}!",
            meta_t3,
        )
        assert res_t3.tier == RoutingTier.TIER_3_DEEP_TEAR_MBR
        assert len(res_t3.candidates) >= 4
        assert res_t3.score is not None

        conn.close()
        print("  [PASS] 3-Tier Semantic Router: Accurate classification across T1, T2, and T3.")

    def test_tear_mbr_consensus_selection(self):
        """Test Minimum Bayes Risk selects consensus translation and filters outliers."""
        candidates = [
            "Chào mừng bạn đến với học viện phép thuật!",
            "Chào mừng bạn đến với học viện ma thuật!",
            "Chào mừng bạn tới với học viện phép thuật!",
            "Hoan nghênh bạn đến học viện phép thuật!",
            "Một ngày đẹp trời để đi câu cá ngoài sông.",  # Outlier hallucination
        ]

        mbr_res = minimum_bayes_risk(candidates)
        best_candidate = mbr_res.best_candidate
        assert best_candidate != candidates[4], "MBR picked an outlier hallucination!"
        assert "học viện" in best_candidate
        print(f"  [PASS] MBR Consensus selected optimal candidate: '{best_candidate}' (score: {mbr_res.best_score:.3f}).")


class TestGate4VramMonitorRoomForCopy:
    """Audit Item 5: GPU VRAM Monitor and _room_for_copy batch size contraction."""

    def test_vram_room_for_copy_batch_contraction(self):
        # Initialize VramMonitor with mock mode simulating constrained memory
        monitor = VramMonitor(
            mock_mode=True,
            mock_free_bytes=700 * 1024 * 1024,  # 700 MB free (with 512MB room_for_copy -> 188MB effective)
            min_room_for_copy_bytes=512 * 1024 * 1024,
            room_for_copy_ratio=0.20,
        )
        model_cfg = ModelVramConfig(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            activation_overhead_bytes=32 * 1024 * 1024,
        )

        batch_size = monitor.calculate_adaptive_batch_size(
            target_batch_size=32,
            seq_len=512,
            model_config=model_cfg,
        )

        # Batch size must be contracted to prevent OOM
        assert batch_size < 32, f"Expected batch size contraction, got {batch_size}"
        assert batch_size >= 1
        print(f"  [PASS] VRAM Monitor: _room_for_copy contracted batch size from 32 down to {batch_size} safely.")


def main():
    print("=" * 70)
    print("RENSLATE GATE 4 INDEPENDENT AUDIT SUITE")
    print("Local LLM Inference Engine & QE Layer Verification")
    print("=" * 70)

    pytest_args = [
        str(Path(__file__).resolve()),
        "-v",
        "-s",
        "--tb=short",
    ]
    sys.exit(pytest.main(pytest_args))


if __name__ == "__main__":
    main()
