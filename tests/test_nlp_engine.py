"""
tests/test_nlp_engine.py
Comprehensive unit test suite for the NLP Engine components:
1. interpolation_lexer.py (renpy.substitutions lexer, 7-step pipeline, invariant placeholders)
2. lcs_segmenter.py (LCS variant alignment, inline branching templates, stitching)
3. constraint_compiler.py (Hybrid constraints: GBNF grammar for N<=5, LogitsProcessor for N>5)
4. tear_mbr.py (TEaR optimization cycle & Minimum Bayes Risk K=4..8 decoding)
"""

import pytest
from typing import List

from renloc_core.llm_engine.interpolation_lexer import (
    TokenType,
    Token,
    PlaceholderMap,
    normalize_flags,
    apply_pipeline,
    ruby_to_renpy,
    renpy_to_ruby,
    tokenize,
    mask_substitutions,
    restore_substitutions,
    validate_placeholders,
    InterpolationLexer,
)
from renloc_core.llm_engine.lcs_segmenter import (
    SegmentKind,
    Segment,
    tokenize_words,
    compute_lcs_indices,
    lcs_similarity,
    segment_pair,
    segment_variants,
    to_template,
    from_template,
    expand_template,
    stitch_variants,
    find_variant_clusters,
    LCSSegmenter,
)
from renloc_core.llm_engine.constraint_compiler import (
    TAG_THRESHOLD,
    ConstraintMode,
    ConstraintSpec,
    GrammarCompiler,
    TagStackLogitsProcessor,
    safe_recover_tags,
    ConstraintCompiler,
    HybridConstraintCompiler,
)
from renloc_core.llm_engine.tear_mbr import (
    compute_chrf,
    compute_lcs_similarity,
    minimum_bayes_risk,
    MBRResult,
    estimate_translation_quality,
    QualityEstimate,
    TEaRConfig,
    TEaRResult,
    TEaRPipeline,
    TEaRMBREngine,
)


# ==============================================================================
# 1. Tests for interpolation_lexer.py
# ==============================================================================

class TestInterpolationLexer:
    def test_normalize_flags_canonical_order(self):
        # 7-step pipeline: r/s -> t -> i -> q -> f -> u -> l -> c
        assert normalize_flags("!cl") == "!lc"
        assert normalize_flags("!lc") == "!lc"
        assert normalize_flags("!it") == "!ti"
        assert normalize_flags("!ti") == "!ti"
        assert normalize_flags("!c!l!t") == "!tlc"
        assert normalize_flags("!u!c") == "!uc"
        assert normalize_flags("!q!u") == "!qu"

    def test_normalize_flags_redundant_exclamation_and_dedup(self):
        # Ignores redundant '!' and deduplicates
        assert normalize_flags("!!ti") == "!ti"
        assert normalize_flags("!t!i") == "!ti"
        assert normalize_flags("!tt!ii") == "!ti"
        assert normalize_flags("!f!f!q") == "!qf"
        assert normalize_flags("") == ""
        assert normalize_flags("!") == ""
        assert normalize_flags("!   ") == ""

    def test_apply_pipeline_order_independence(self):
        # !cl and !lc must yield identical results: lower then capitalize
        val = "hELLO wORLD"
        res_cl = apply_pipeline(val, "!cl")
        res_lc = apply_pipeline(val, "!lc")
        assert res_cl == res_lc == "Hello world"

    def test_apply_pipeline_all_stages(self):
        # r/s -> repr or str
        assert apply_pipeline(123, "!s") == "123"
        assert apply_pipeline("hello", "!r") == "'hello'"

        # t -> translate
        def mock_trans(s):
            return f"trans_{s}"
        assert apply_pipeline("cat", "!t", translate_fn=mock_trans) == "trans_cat"

        # i -> recursive interpolation
        def mock_interp(s):
            return s.replace("{x}", "injected")
        assert apply_pipeline("{x}", "!i", interpolate_fn=mock_interp) == "injected"

        # q -> quote text tags
        assert apply_pipeline("a {b} c [d]", "!q") == "a {{b} c [[d]"

        # f -> say_menu_text_filter
        def mock_filter(s):
            return f"filtered({s})"
        assert apply_pipeline("raw", "!f", filter_fn=mock_filter) == "filtered(raw)"

        # u / l / c
        assert apply_pipeline("hello", "!u") == "HELLO"
        assert apply_pipeline("HELLO", "!l") == "hello"
        assert apply_pipeline("hello world", "!c") == "Hello world"

    def test_ruby_text_conversions(self):
        # Fullwidth pipe: 【dưới｜trên】
        orig = "Người đó là 【học sinh｜sinh viên】 của trường."
        renpy_style = ruby_to_renpy(orig)
        assert renpy_style == "Người đó là {rb}học sinh{/rb}{rt}sinh viên{/rt} của trường."
        assert renpy_to_ruby(renpy_style) == orig

        # Halfwidth pipe: 【base|ruby】
        half = "Ký tự 【漢｜かん】."
        assert ruby_to_renpy(half) == "Ký tự {rb}漢{/rb}{rt}かん{/rt}."

    def test_escape_sequences_and_spaces(self):
        # \  preserves extra whitespace
        text = r"One \ \ Two"
        masked, pmap = mask_substitutions(text)
        assert "<T0>" in masked and "<T1>" in masked
        restored = restore_substitutions(masked, pmap)
        assert restored == text

        # \n and \\
        text2 = r"Dòng 1\nDòng 2\\Thư mục"
        masked2, pmap2 = mask_substitutions(text2)
        restored2 = restore_substitutions(masked2, pmap2)
        assert restored2 == text2

        # Escaped brackets {{ and [[
        text3 = "Literal {{tag}} and [[var]]"
        masked3, pmap3 = mask_substitutions(text3)
        restored3 = restore_substitutions(masked3, pmap3)
        assert restored3 == text3

    def test_invariant_placeholders_1to1_mapping(self):
        text = (
            r"Xin chào, \  [player_name!ti]! Bạn có muốn {b}bắt đầu{/b} "
            "với giá [coins!!s] vàng không?"
        )
        masked, pmap = mask_substitutions(text)

        # Invariant placeholder format check
        assert "<T0>" in masked         # \ 
        assert "<V0!ti>" in masked      # [player_name!ti]
        assert "<T1>" in masked         # {b}
        assert "<T2>" in masked         # {/b}
        assert "<V1!s>" in masked       # [coins!!s] normalized to !s

        # 1-1 deterministic restoration
        restored = restore_substitutions(masked, pmap)
        assert restored == text

        # Placeholder validation check
        is_valid, missing, extra = validate_placeholders(masked, pmap)
        assert is_valid is True
        assert len(missing) == 0
        assert len(extra) == 0

    def test_validate_placeholders_missing_and_extra(self):
        text = "Hello [name!u] and {i}welcome{/i}!"
        masked, pmap = mask_substitutions(text)
        # Simulate LLM dropping <T1> and hallucinating <T99>
        corrupted = masked.replace("<T1>", "") + " <T99>"

        is_valid, missing, extra = validate_placeholders(corrupted, pmap)
        assert is_valid is False
        assert "<T1>" in missing
        assert "<T99>" in extra

    def test_nested_brackets_and_format_specifiers(self):
        # Nested brackets inside variable: [items[0].name!ti]
        # Format specifier: [score:04d!u]
        text = "Nhận được [items[0].name!ti] với số điểm [score:04d!u] và [player.stats['hp']:02d]!"
        masked, pmap = mask_substitutions(text)
        assert "<V0!ti>" in masked
        assert "<V1!u>" in masked
        assert "<V2>" in masked
        restored = restore_substitutions(masked, pmap)
        assert restored == text

    def test_interpolation_lexer_convenience_class(self):
        text = "Test [hero!cl] {b}tag{/b}"
        masked, pmap = InterpolationLexer.mask(text)
        assert "<V0!lc>" in masked
        restored = InterpolationLexer.restore(masked, pmap)
        assert restored == text

    def test_fuzz_10000_nested_tags_and_variables(self):
        import random
        tags_pool = ['{b}', '{/b}', '{i}', '{/i}', '{color=#fff}', '{/color}', '{w}', '{p}', r'\ ', r'\n', '【đêm｜dạ】']
        vars_pool = ['[player_name]', '[player_name!ti]', '[hero!u]', '[coins!!s]', '[title!cl]']
        text_pool = ['Xin chào', 'hôm nay', 'chúng ta', 'bắt đầu', 'cuộc hành trình', 'đầy thử thách', 'tại Ren\'Py', 'này.']

        random.seed(42)
        for _ in range(10000):
            elements = random.sample(text_pool, k=random.randint(2, 4))
            elements += random.sample(tags_pool, k=random.randint(1, 3))
            elements += random.sample(vars_pool, k=random.randint(1, 2))
            random.shuffle(elements)
            sentence = " ".join(elements)

            masked, pmap = mask_substitutions(sentence)
            valid, missing, extra = validate_placeholders(masked, pmap)
            assert valid is True and len(missing) == 0 and len(extra) == 0

            restored = restore_substitutions(masked, pmap)
            assert restored == sentence



# ==============================================================================
# 2. Tests for lcs_segmenter.py
# ==============================================================================

class TestLCSSegmenter:
    def test_lcs_similarity(self):
        s1 = "Chào mừng anh đến với thành phố tương lai."
        s2 = "Chào mừng chị đến với thành phố tương lai."
        sim = lcs_similarity(s1, s2)
        assert 0.85 < sim < 1.0

        # Exact match
        assert lcs_similarity(s1, s1) == 1.0
        # Empty
        assert lcs_similarity("", "abc") == 0.0

    def test_segment_pair_dialogue_variants(self):
        v1 = "Chào anh, em rất vui vì được gặp lại anh hôm nay!"
        v2 = "Chào chị, em rất vui vì được gặp lại chị hôm nay!"

        segments = segment_pair(v1, v2)
        assert len(segments) == 5

        assert segments[0].kind == SegmentKind.COMMON
        assert segments[0].text == "Chào "

        assert segments[1].kind == SegmentKind.BRANCH
        assert segments[1].branches == ("anh", "chị")

        assert segments[2].kind == SegmentKind.COMMON
        assert segments[2].text == ", em rất vui vì được gặp lại "

        assert segments[3].kind == SegmentKind.BRANCH
        assert segments[3].branches == ("anh", "chị")

        assert segments[4].kind == SegmentKind.COMMON
        assert segments[4].text == " hôm nay!"

    def test_to_and_from_template(self):
        v1 = "Chào anh, em rất vui được gặp lại anh hôm nay!"
        v2 = "Chào chị, em rất vui được gặp lại chị hôm nay!"
        segments = segment_pair(v1, v2)

        template = to_template(segments)
        assert template == "Chào {anh|chị}, em rất vui được gặp lại {anh|chị} hôm nay!"

        parsed_segs = from_template(template)
        assert len(parsed_segs) == len(segments)
        for s_orig, s_parsed in zip(segments, parsed_segs):
            assert s_orig.kind == s_parsed.kind
            if s_orig.kind == SegmentKind.COMMON:
                assert s_orig.text == s_parsed.text
            else:
                assert s_orig.branches == s_parsed.branches

    def test_expand_and_stitch_variants(self):
        template = "Tôi {rất |hoàn toàn }đồng ý với {anh|chị}."
        expanded = expand_template(template)
        assert len(expanded) == 2
        assert expanded[0] == "Tôi rất đồng ý với anh."
        assert expanded[1] == "Tôi hoàn toàn đồng ý với chị."

        segs = from_template(template)
        stitched = stitch_variants(segs)
        assert stitched == expanded

    def test_multi_variants_alignment(self):
        v1 = "Chào anh."
        v2 = "Chào chị."
        v3 = "Chào bạn."
        segs = segment_variants([v1, v2, v3])
        template = to_template(segs)
        assert template == "Chào {anh|chị|bạn}."

        expanded = expand_template(template)
        assert expanded == [v1, v2, v3]

    def test_find_variant_clusters(self):
        dialogues = [
            "Anh có muốn đi dạo cùng em không?",
            "Chị có muốn đi dạo cùng em không?",
            "Hôm nay trời đổ mưa bão lớn quá.",
            "Ngày mai trời đổ mưa bão lớn quá.",
            "Một câu thoại hoàn toàn độc lập khác."
        ]
        clusters = find_variant_clusters(dialogues, min_similarity=0.7)
        assert len(clusters) == 3
        # First cluster has the 2 pronoun variants
        assert len(clusters[0]) == 2
        # Second cluster has the weather variants
        assert len(clusters[1]) == 2
        # Third cluster is independent
        assert len(clusters[2]) == 1

    def test_lcs_segmenter_convenience_class(self):
        v1 = "Đi thôi anh!"
        v2 = "Đi thôi chị!"
        segs = LCSSegmenter.segment(v1, v2)
        t = LCSSegmenter.to_template(segs)
        assert t == "Đi thôi {anh|chị}!"
        exp = LCSSegmenter.expand(t)
        assert exp == [v1, v2]


# ==============================================================================
# 3. Tests for constraint_compiler.py
# ==============================================================================

class TestConstraintCompiler:
    def test_hybrid_decision_threshold(self):
        compiler = ConstraintCompiler(threshold=TAG_THRESHOLD)

        # N <= 5 -> GBNF_GRAMMAR
        spec_small = compiler.compile(tags=["<T0>", "<T1>", "<V0!ti>"])
        assert spec_small.mode == ConstraintMode.GBNF_GRAMMAR
        assert spec_small.grammar is not None
        assert spec_small.processor is None

        # N > 5 -> LOGITS_PROCESSOR
        tags_6 = [f"<T{i}>" for i in range(6)]
        spec_large = compiler.compile(tags=tags_6)
        assert spec_large.mode == ConstraintMode.LOGITS_PROCESSOR
        assert spec_large.processor is not None
        assert spec_large.grammar is None

    def test_gbnf_grammar_syntax_and_paired_tags(self):
        grammar = GrammarCompiler.compile_gbnf(
            tags=["<T0>", "{w}"],
            paired_tags=[("{b}", "{/b}")],
        )
        assert 'paired_0 ::= "{b}" content "{/b}"' in grammar
        assert 'tag_0 ::= "<T0>"' in grammar
        assert 'tag_1 ::= "{w}"' in grammar
        assert "standalone_tag ::= tag_0 | tag_1" in grammar
        assert "root ::= content" in grammar

    def test_ebnf_grammar_generation(self):
        ebnf = GrammarCompiler.compile_ebnf(
            tags=["<T0>"],
            paired_tags=[("{i}", "{/i}")],
        )
        assert 'paired_0 = "{i}", content, "{/i}" ;' in ebnf
        assert 'tag = "<T0>" ;' in ebnf
        assert "root = content ;" in ebnf

    def test_tag_stack_logits_processor_eos_masking(self):
        proc = TagStackLogitsProcessor(
            expected_tags=["{b}", "{/b}"],
            eos_token_id=5,
        )

        # Initially stack empty, EOS token is not masked
        scores = [1.0] * 10
        proc([1, 2], scores)
        assert scores[5] == 1.0

        # Push open tag {b}
        proc.on_tag_opened("{b}")
        assert len(proc.tag_stack) == 1
        proc([1, 2], scores)
        assert scores[5] == float("-inf"), "EOS must be masked to -inf when tags are open!"

        # Pop closing tag {/b}
        closed_ok = proc.on_tag_closed("{/b}")
        assert closed_ok is True
        assert len(proc.tag_stack) == 0

    def test_tag_stack_logits_processor_deadlock_safe_recovery(self):
        proc = TagStackLogitsProcessor(
            expected_tags=["{b}", "{/b}"],
            paired_tags_map={"{b}": "{/b}"},
            tag_to_tokens_map={"{/b}": [42]},
        )
        proc.on_tag_opened("{b}")

        # Simulate all logits becoming -inf (deadlock condition)
        all_inf_scores = [float("-inf")] * 50
        proc([1, 2], all_inf_scores)

        # Must recover by unblocking the closing tag token 42
        assert all_inf_scores[42] == 0.0
        assert proc.deadlock_recovered_count == 1

    def test_safe_recover_tags_unclosed_tags(self):
        # Open tag {b} and {i} without closing
        text = "Chào mừng bạn đến với {b}thế giới {i}kỳ diệu"
        recovered = safe_recover_tags(text, expected_tags=[])
        # Auto-closes in LIFO order: {/i}{/b}
        assert recovered == "Chào mừng bạn đến với {b}thế giới {i}kỳ diệu{/i}{/b}"

    def test_safe_recover_tags_orphan_closing_tag(self):
        # Orphan {/b} without opening tag
        text = "Xin chào {/b}các bạn!"
        recovered = safe_recover_tags(text, expected_tags=[])
        assert recovered == "Xin chào các bạn!"

    def test_safe_recover_tags_missing_required_placeholders(self):
        # Translation missed <T0> and <V1!ti>
        text = "Xin chào các bạn!"
        recovered = safe_recover_tags(text, expected_tags=["<T0>", "<V1!ti>"])
        assert "<T0>" in recovered
        assert "<V1!ti>" in recovered

    def test_hybrid_constraint_compiler_alias(self):
        compiler = HybridConstraintCompiler(threshold=3)
        spec = compiler.compile(tags=["<T0>", "<T1>", "<T2>", "<T3>"])
        assert spec.mode == ConstraintMode.LOGITS_PROCESSOR


# ==============================================================================
# 4. Tests for tear_mbr.py
# ==============================================================================

class TestTEaRMBR:
    def test_compute_chrf_properties(self):
        t1 = "Xin chào bạn bè thân thiết!"
        t2 = "Xin chào những người bạn thân!"
        score = compute_chrf(t1, t2)
        assert 0.4 < score < 0.9

        # Identity
        assert compute_chrf(t1, t1) == 1.0
        # Empty
        assert compute_chrf("", t1) == 0.0

    def test_compute_lcs_utility(self):
        s1 = "Hello world!"
        s2 = "Hello there world!"
        score = compute_lcs_similarity(s1, s2)
        assert score >= 0.8
        assert compute_lcs_similarity(s1, s1) == 1.0

    def test_minimum_bayes_risk_centroid_selection(self):
        # 5 candidates: 4 similar translations and 1 outlier hallucination
        candidates = [
            "Chào bạn, hôm nay trời nắng đẹp.",
            "Chào bạn, thời tiết hôm nay thật đẹp.",
            "Chào cậu, hôm nay trời nắng rất đẹp.",
            "Chào bạn, bầu trời hôm nay rất đẹp.",
            "Hãy nhấp vào liên kết này để nhận quà tặng miễn phí ngay bây giờ!" # Hallucination outlier
        ]
        mbr = minimum_bayes_risk(candidates, utility_metric="chrf")
        assert mbr.sample_size == 5
        # The outlier must have the lowest score and not be chosen
        assert mbr.best_index != 4
        assert mbr.candidate_scores[4] < 0.20
        assert mbr.best_score > 0.45
        assert mbr.best_score > 3.0 * mbr.candidate_scores[4]

    def test_estimate_translation_quality_penalties(self):
        # Tag penalty
        est_missing_tag = estimate_translation_quality(
            source_text="Hello <T0>",
            translation="Xin chào",
            expected_placeholders=["<T0>"],
        )
        assert est_missing_tag.is_acceptable is False
        assert any("missing_placeholders:<T0>" in iss for iss in est_missing_tag.issues)

        # Repetition penalty
        est_rep = estimate_translation_quality(
            source_text="Normal sentence",
            translation="Xin chào bạn bạn bạn bạn bạn bạn bạn bạn bạn bạn",
        )
        assert est_rep.is_acceptable is False
        assert "degenerate_repetition" in est_rep.issues

        # High quality
        est_good = estimate_translation_quality(
            source_text="Hello world! <T0>",
            translation="Xin chào thế giới! <T0>",
            expected_placeholders=["<T0>"],
        )
        assert est_good.is_acceptable is True
        assert est_good.score >= 0.85

    def test_tear_pipeline_refinement_cycle(self):
        config = TEaRConfig(
            k_candidates=4,
            qe_threshold=0.85,
            max_refinements=2,
            auto_recover_tags=True,
        )
        pipeline = TEaRPipeline(config=config)

        # Mock translate emits candidates without the required tag
        def mock_translate(src: str, k: int) -> List[str]:
            return [
                "Xin chào thế giới",
                "Xin chào các bạn",
                "Chào mừng thế giới",
                "Chào bạn",
            ]

        # Mock refine fixes the missing tag
        def mock_refine(src: str, current: str, issues: List[str]) -> str:
            assert any("missing_placeholders" in iss for iss in issues)
            return f"{current} <T0>"

        result = pipeline.run(
            source_text="Hello world <T0>",
            translate_fn=mock_translate,
            refine_fn=mock_refine,
            expected_tags=["<T0>"],
        )

        assert result.refinements_count == 1
        assert "<T0>" in result.final_translation
        assert result.final_score > result.initial_score
        assert len(result.history) == 2  # initial + 1 refine

    def test_tear_mbr_engine_static_helpers(self):
        cands = ["Câu một", "Câu hai", "Câu một"]
        mbr = TEaRMBREngine.select_mbr(cands)
        assert mbr.best_candidate == "Câu một"
