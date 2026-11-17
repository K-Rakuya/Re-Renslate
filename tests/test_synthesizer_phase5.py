"""
tests/test_synthesizer_phase5.py
Comprehensive unit test suite for Phase 5 Synthesizer components:
  1. incremental_matcher.py: 4-Tier matching (Tier 1 Exact, Tier 2 Structural, Tier 3 Fuzzy, Tier 4 New).
  2. package_synthesizer.py: 4-space indent, 0 tabs, dialogue translate blocks, UI strings, font group overrides.
  3. runtime_hook.py: Revertable* rollback, filter idempotency, and 90-frame storm circuit breaker.
"""
import tempfile
from pathlib import Path
import pytest

from renloc_core.static_graph.dialogue_normalizer import DialogueEntity, DialogueForm
from renloc_core.synthesizer import (
    MatchTier,
    IncrementalMatchResult,
    IncrementalMatcher,
    levenshtein_similarity,
    TranslationItem,
    UIStringTranslationItem,
    PackageSynthesizer,
    StormCircuitBreaker,
    RenlocCircuitBreaker,
    RuntimeHookManager,
    generate_runtime_hook_rpy,
)


# =========================================================================
# 1. Incremental Matcher Tests (4 Tiers)
# =========================================================================

class TestIncrementalMatcher:
    def test_levenshtein_similarity_metrics(self):
        assert levenshtein_similarity("hello", "hello") == 1.0
        assert levenshtein_similarity("", "") == 1.0
        assert levenshtein_similarity("kitten", "sitting") > 0.50
        assert levenshtein_similarity("abc", "xyz") == 0.0

    def test_4_tier_matching_workflow(self):
        matcher = IncrementalMatcher(fuzzy_threshold=0.70)

        # Old translation corpus
        old_e1 = DialogueEntity(
            entity_id="old_1",
            who="sylvia",
            what="Welcome to the ancient ruins!",
            form=DialogueForm.EXPLICIT,
            filename="script.rpy",
            linenumber=10,
            block_depth=0,
            translation_id="tl_exact_001",
        )
        old_e2 = DialogueEntity(
            entity_id="old_2",
            who="sylvia",
            what="Are you prepared for the journey?",
            form=DialogueForm.EXPLICIT,
            filename="script.rpy",
            linenumber=12,
            block_depth=0,
            translation_id="tl_struct_002",
            attributes=("serious",),
        )
        old_e3 = DialogueEntity(
            entity_id="old_3",
            who=None,
            what="The cold wind whispers through stone walls.",
            form=DialogueForm.NARRATION,
            filename="script.rpy",
            linenumber=15,
            block_depth=0,
            translation_id="tl_fuzzy_003",
        )

        old_corpus = {
            "tl_exact_001": (old_e1, "Chào mừng đến với tàn tích cổ xưa!"),
            "tl_struct_002": (old_e2, "Bạn đã sẵn sàng cho hành trình chưa?"),
            "tl_fuzzy_003": (old_e3, "Cơn gió lạnh thì thầm qua những bức tường đá."),
        }

        # New version units:
        # 1. Exact match (same ID)
        new_u1 = DialogueEntity(
            entity_id="new_1",
            who="sylvia",
            what="Welcome to the ancient ruins!",
            form=DialogueForm.EXPLICIT,
            filename="script.rpy",
            linenumber=10,
            block_depth=0,
            translation_id="tl_exact_001",
        )

        # 2. Structural match (same who + what, but different attributes -> new ID)
        new_u2 = DialogueEntity(
            entity_id="new_2",
            who="sylvia",
            what="Are you prepared for the journey?",
            form=DialogueForm.EXPLICIT,
            filename="script.rpy",
            linenumber=14,
            block_depth=0,
            translation_id="tl_struct_modified_hash",
            attributes=("happy", "smile"), # modified attributes!
        )

        # 3. Fuzzy match (minor typo or wording update in narration)
        new_u3 = DialogueEntity(
            entity_id="new_3",
            who=None,
            what="The cold wind whispers through the stone walls.", # added 'the'
            form=DialogueForm.NARRATION,
            filename="script.rpy",
            linenumber=18,
            block_depth=0,
            translation_id="tl_fuzzy_different_hash",
        )

        # 4. Completely new unit
        new_u4 = DialogueEntity(
            entity_id="new_4",
            who="eileen",
            what="I have arrived from another land.",
            form=DialogueForm.EXPLICIT,
            filename="script.rpy",
            linenumber=30,
            block_depth=0,
            translation_id="tl_completely_new",
        )

        matches = matcher.match_units([new_u1, new_u2, new_u3, new_u4], old_corpus)

        assert len(matches) == 4

        # Check Tier 1
        m1 = matches[0]
        assert m1.tier == MatchTier.TIER_1_EXACT_ID
        assert m1.confidence == 1.0
        assert m1.target_translation == "Chào mừng đến với tàn tích cổ xưa!"

        # Check Tier 2
        m2 = matches[1]
        assert m2.tier == MatchTier.TIER_2_STRUCTURAL_AST
        assert m2.confidence == 0.95
        assert m2.target_translation == "Bạn đã sẵn sàng cho hành trình chưa?"

        # Check Tier 3
        m3 = matches[2]
        assert m3.tier == MatchTier.TIER_3_FUZZY_SIMILARITY
        assert m3.confidence >= 0.70
        assert m3.target_translation == "Cơn gió lạnh thì thầm qua những bức tường đá."

        # Check Tier 4
        m4 = matches[3]
        assert m4.tier == MatchTier.TIER_4_NEW_TASK
        assert m4.target_translation is None
        assert m4.confidence == 0.0


# =========================================================================
# 2. Package Synthesizer Tests
# =========================================================================

class TestPackageSynthesizer:
    def test_format_dialogue_block_and_tab_forbiddance(self):
        synth = PackageSynthesizer(language="vietnamese")
        item = TranslationItem(
            translation_id="start_d9ca682b",
            original_statement_code='s "Welcome to the old world!"',
            translated_text="Chào mừng bạn đến với thế giới cổ xưa!",
            who="s",
            filename="script.rpy",
            linenumber=36,
        )

        formatted = synth.format_dialogue_block(item)
        assert "translate vietnamese start_d9ca682b:" in formatted
        assert '# s "Welcome to the old world!"' in formatted
        assert 's "Chào mừng bạn đến với thế giới cổ xưa!"' in formatted

        # Check indentation: exactly 4 spaces, 0 tabs
        assert "\t" not in formatted
        for line in formatted.splitlines():
            if line.startswith(" "):
                assert line.startswith("    ")

    def test_format_strings_file(self):
        synth = PackageSynthesizer(language="vietnamese")
        items = [
            UIStringTranslationItem(original_text="Start Game", translated_text="Bắt đầu", filename="screens.rpy", linenumber=96),
            UIStringTranslationItem(original_text="Quit", translated_text="Thoát", filename="screens.rpy", linenumber=99),
        ]

        text = synth.format_strings_file(items)
        assert "translate vietnamese strings:" in text
        assert 'old "Start Game"' in text
        assert 'new "Bắt đầu"' in text
        assert "\t" not in text

    def test_export_package_filesystem(self):
        synth = PackageSynthesizer(language="vietnamese")
        dlg = [
            TranslationItem(
                translation_id="act1_001",
                original_statement_code='e "Hello"',
                translated_text="Xin chào",
                who="e",
                filename="script.rpy",
                linenumber=10,
            )
        ]
        ui = [
            UIStringTranslationItem(original_text="Preferences", translated_text="Cài đặt", filename="screens.rpy", linenumber=50)
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_root = Path(tmp_dir) / "game"
            created = synth.export_package(out_root, dlg, ui)

            assert "script.rpy" in created
            assert "common.rpy" in created
            assert "00_renloc_fonts.rpy" in created

            # Verify files on disk have zero tabs
            for name, path in created.items():
                if path.suffix == ".rpy":
                    content = path.read_text(encoding="utf-8")
                    assert "\t" not in content, f"Tab found in {name}!"
                    assert "translate vietnamese" in content or "FontGroup" in content


# =========================================================================
# 3. Runtime Hook & 90-Frame Storm Circuit Breaker Tests
# =========================================================================

class TestRuntimeHookAndBreaker:
    def test_revertable_state_and_idempotency(self):
        manager = RuntimeHookManager()
        manager.dynamic_context["Player"] = "Người chơi"

        # Pillar 2: Idempotency (calling filter 5 times yields identical string)
        text = "Hello, Player! Welcome back, Player!"
        first_pass = manager.filter_text(text)
        assert first_pass == "Hello, Người chơi! Welcome back, Người chơi!"

        for _ in range(5):
            assert manager.filter_text(text) == first_pass

    def test_rollback_checkpoint_stability(self):
        manager = RuntimeHookManager()
        manager.dynamic_context["mystery_name"] = "???"
        assert manager.filter_text("Who is mystery_name?") == "Who is ????"

        # Save checkpoint (before reveal)
        manager.checkpoint()

        # Update dynamic name after reveal
        manager.dynamic_context["mystery_name"] = "Sylvia"
        assert manager.filter_text("Who is mystery_name?") == "Who is Sylvia?"

        # Simulate player rolling back
        manager.rollback()
        assert manager.filter_text("Who is mystery_name?") == "Who is ????"

    def test_90_frame_storm_circuit_breaker(self):
        breaker = StormCircuitBreaker(max_consecutive_errors=90)
        assert breaker.tripped is False

        # Simulate 89 consecutive errors -> breaker still closed
        for _ in range(89):
            breaker.record_error("test error")
            assert breaker.tripped is False

        # 90th error -> breaker trips!
        breaker.record_error("90th error")
        assert breaker.tripped is True
        assert breaker.trip_count == 1

        # While tripped, manager bypasses filtering and returns raw text safely
        manager = RuntimeHookManager()
        manager.breaker = breaker
        raw_input = "Recursion loop error !i"
        assert manager.filter_text(raw_input) == raw_input

        # Reset breaker
        breaker.reset()
        assert breaker.tripped is False
        assert breaker.consecutive_errors == 0

    def test_generate_runtime_hook_rpy_formatting(self):
        rpy_code = generate_runtime_hook_rpy(breaker_threshold=90)
        assert "init -990 python:" in rpy_code
        assert "class _RenlocStormCircuitBreaker(object):" in rpy_code
        assert "threshold=90" in rpy_code
        assert "config.say_menu_text_filter" in rpy_code
        assert "\t" not in rpy_code, "Tabs strictly forbidden in Ren'Py runtime hook!"
