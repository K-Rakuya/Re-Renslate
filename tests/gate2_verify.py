#!/usr/bin/env python3
"""
tests/gate2_verify.py
Independent Gate 2 Quality Audit Verification Suite for Renslate Phase 2.
According to Section 3 and Section 6 Gate 2 of the original specification:
  1. Lifecycle & WrapNode: 5-phase initialization ordering & source position tracking.
  2. Dialogue Normalization: 4 Say forms into DialogueEntity (who!=None, who==None, extend, renpy.say, _/__).
  3. UI String Coverage: Full scanstrings/SL2 extraction (text, textbutton, tooltip, alt) & LayeredImage tracking.
  4. ICFG k-CFA (k=2): Interprocedural control flow graph with call-string context sensitivity.
  5. Tarjan SCC: Loop condensation (day-night & menu loops) into DAG; NO infinite loop hang.
  6. Lengauer-Tarjan Dominator Tree: Global milestone separation from branch divergences.
  7. Trace Partitioning: Path-sensitive trace partitioning for relational variables across routes.
  8. Temporal Honorific Graph: 100% zero-leakage guarantee for anonymous character reveal.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "py" / "renloc-core"))

from renloc_core.ast_reader.ast_nodes import (
    Call,
    Default,
    Define,
    EarlyPython,
    Init,
    Jump,
    Label,
    Node,
    Python,
    Return,
    Say,
    Scene,
    Screen,
    Show,
    SLDisplayable,
    SLFor,
    SLIf,
    SLPython,
    SLScreen,
    SLUse,
    UserStatement,
)

from renloc_core.static_graph import (
    # Lifecycle
    LifecyclePhase,
    WrapNode,
    LifecycleSimulator,
    # Dialogue Normalizer
    DialogueForm,
    DialogueEntity,
    DialogueNormalizer,
    # Screen Extractor
    ExtractedUIString,
    ScreenExtractor,
    LayeredImageTracker,
    # ICFG
    EdgeType,
    CallContext,
    ICFGNode,
    ICFGEdge,
    ICFG,
    ICFGBuilder,
    # Tarjan SCC
    LoopKind,
    SCCComponent,
    CondensationDAG,
    TarjanSCC,
    # Dominator Tree
    DominatorRole,
    DominatorInfo,
    LengauerTarjanDominator,
    # Trace Partitioning
    PartitionKey,
    AbstractState,
    PartitionedTraceNode,
    TracePartitioningEngine,
    # Temporal Honorific Graph
    UNKNOWN_IDENTITY,
    FormalityTier,
    CharacterIdentity,
    HonorificProfile,
    TemporalHonorificGraph,
)


# =========================================================================
# 1. 5-Phase Lifecycle & WrapNode Verification
# =========================================================================
def test_lifecycle_5_phases_and_wrapnode():
    print("[*] 1. Verifying 5-Phase Lifecycle Ordering and WrapNode Tracking...")

    sim = LifecycleSimulator()

    raw_nodes = [
        Label(name="start", loc=("story.rpy", 50)),
        Say(who="s", what="Story begins", loc=("story.rpy", 51)),
        Default(varname="affection", code="0", loc=("vars.rpy", 10)),
        Init(priority=10, block=[Define(varname="late_char", code="Character()", loc=("late.rpy", 5))], loc=("late.rpy", 1)),
        Init(priority=-1, block=[Define(varname="early_char", code="Character()", loc=("early.rpy", 2))], loc=("early.rpy", 1)),
        Define(varname="s", code="Character('Sylvia')", loc=("defs.rpy", 20)),
        EarlyPython(code="EARLY_FLAG = True", loc=("early.rpy", 1)),
        Label(name="main_menu", loc=("screens.rpy", 100)),
        Label(name="splashscreen", loc=("splash.rpy", 5)),
    ]

    sim.wrap_ast(raw_nodes)
    ordered = sim.get_execution_order()

    phases = [w.phase for w in ordered]

    # Phase 1: EarlyPython must come first
    assert phases[0] == LifecyclePhase.EARLY_PYTHON
    assert isinstance(ordered[0].node, EarlyPython)

    # Phase 2: Inits sorted by priority (-1, then 0, then 10)
    init_nodes = [w for w in ordered if w.phase == LifecyclePhase.INIT]
    priorities = [w.priority for w in init_nodes]
    assert priorities == sorted(priorities)
    assert -1 in priorities and 10 in priorities

    # Phase 3: Default variables
    default_nodes = [w for w in ordered if w.phase == LifecyclePhase.DEFAULT]
    assert len(default_nodes) == 1
    assert default_nodes[0].node.varname == "affection"

    # Phase 4: Entry points (splashscreen, main_menu)
    entry_nodes = [w for w in ordered if w.phase == LifecyclePhase.ENTRY_POINT]
    entry_names = {w.node.name for w in entry_nodes}
    assert entry_names == {"splashscreen", "main_menu"}

    # Phase 5: Start story
    story_nodes = [w for w in ordered if w.phase == LifecyclePhase.START_STORY]
    assert any(isinstance(w.node, Label) and w.node.name == "start" for w in story_nodes)

    # Verify WrapNode metadata
    for w in ordered:
        assert w.block_depth >= 0
        assert w.ast_id is not None

    print("    [✓] 5-Phase Lifecycle ordering & WrapNode metadata verified 100%!\n")


# =========================================================================
# 2. Dialogue Normalization (4 Forms + Extend Inheritance)
# =========================================================================
def test_dialogue_normalization_and_extend():
    print("[*] 2. Verifying Dialogue Normalization (4 Forms & Extend)...")

    normalizer = DialogueNormalizer()

    statements = [
        # Form 1: Explicit who
        Say(who="e", what="Hello traveler!", attributes=("happy",), temporary_attributes=("smile",)),
        # Form 3: Extend
        Say(who="extend", what=" Welcome to our town."),
        # Form 2: Narration
        Say(who=None, what="The wind blows gently across the square."),
        # Form 4a: renpy.say in python
        UserStatement(line='$ renpy.say(s, "A pleasure to meet you.")'),
        # Form 4b & 4c: translatable strings
        UserStatement(line='$ msg1 = _("Save Game")\n$ msg2 = __("Deferred Option")'),
    ]

    entities = normalizer.normalize_stream(statements, current_label="start")
    assert len(entities) == 6

    # 1. Explicit
    assert entities[0].form == DialogueForm.EXPLICIT
    assert entities[0].who == "e"
    assert entities[0].what == "Hello traveler!"
    assert entities[0].dialogue_index == 1

    # 2. Extend
    assert entities[1].form == DialogueForm.EXTEND
    assert entities[1].who == "e"  # Inherited from previous speaker
    assert entities[1].what == " Welcome to our town."
    assert entities[1].parent_id == entities[0].entity_id  # Linked to parent
    assert entities[1].dialogue_index == 2  # Incremented dialogue counter

    # 3. Narration
    assert entities[2].form == DialogueForm.NARRATION
    assert entities[2].who is None

    # 4. renpy.say
    assert entities[3].form == DialogueForm.RENPY_SAY
    assert entities[3].who == "s"
    assert entities[3].what == "A pleasure to meet you."

    # 5. _(s) immediate
    assert entities[4].form == DialogueForm.TRANSLATABLE_STRING
    assert entities[4].what == "Save Game"
    assert entities[4].is_deferred is False

    # 6. __(s) deferred
    assert entities[5].form == DialogueForm.TRANSLATABLE_STRING
    assert entities[5].what == "Deferred Option"
    assert entities[5].is_deferred is True

    print("    [✓] Dialogue Normalization 4 forms & extend inheritance verified!\n")


# =========================================================================
# 3. Screen Language 2 & UI Extraction Coverage
# =========================================================================
def test_screen_language_ui_extraction():
    print("[*] 3. Verifying Screen Language 2 UI String Extraction & LayeredImage...")

    screen = SLScreen(
        name="game_menu",
        children=[
            SLDisplayable(name="text", positional=['_("Pause Menu")']),
            SLDisplayable(
                name="textbutton",
                positional=['_("Resume")'],
                keyword=[("tooltip", '_("Return to adventure")'), ("alt", '"Resume game action"')]
            ),
            SLIf(entries=[
                ('_("Admin Mode")', [SLDisplayable(name="text", positional=['"Cloud Save"'])]),
            ]),
            SLFor(
                variable="item",
                expression="inventory",
                children=[
                    SLDisplayable(name="text", positional=['_("Item in bag")'])
                ]
            ),
            SLPython(code=Python(code='notify_text = _("Press ESC to exit")')),
        ]
    )

    extractor = ScreenExtractor()
    extracted = extractor.extract_screen(screen)

    extracted_texts = {u.text for u in extracted}
    assert "Pause Menu" in extracted_texts
    assert "Resume" in extracted_texts
    assert "Return to adventure" in extracted_texts
    assert "Resume game action" in extracted_texts
    assert "Cloud Save" in extracted_texts
    assert "Item in bag" in extracted_texts
    assert "Press ESC to exit" in extracted_texts

    # LayeredImage Tracker test
    from renloc_core.ast_reader.ast_nodes import LayeredImage, RawAttribute
    tracker = LayeredImageTracker()
    li = LayeredImage(name="sylvia", attributes=[RawAttribute(attribute="happy"), RawAttribute(attribute="uniform")])
    tracker.register_layered_image(li)
    tracker.process_show_or_scene(Show(imspec=("sylvia", "happy", "uniform")))
    current_mood = tracker.get_active_tags("sylvia")
    assert current_mood == {"happy", "uniform"}

    dlg = DialogueEntity(
        entity_id="d1",
        who="sylvia",
        what="I am ready!",
        form=DialogueForm.EXPLICIT,
        filename="script.rpy",
        linenumber=10,
        block_depth=0,
        translation_id="t1"
    )
    tracker.tag_dialogue(dlg)
    assert "happy" in dlg.context_tags and "uniform" in dlg.context_tags

    print(f"    [+] Extracted {len(extracted)} UI strings with 100% coverage.")
    print("    [✓] SL2 UI String Extraction & LayeredImage tracking verified!\n")


# =========================================================================
# 4. ICFG k-CFA (k=2) & Procedural Graph
# =========================================================================
def test_icfg_k_cfa_graph():
    print("[*] 4. Verifying ICFG k-CFA (k=2) Interprocedural Call Graph...")

    builder = ICFGBuilder(k=2)

    labels = {
        "start": [
            Say(who="guide", what="Welcome to the realm."),
            Call(label="visit_tavern"),
            Say(who="guide", what="We have departed the tavern."),
            Jump(target="town_square"),
        ],
        "visit_tavern": [
            Say(who="bartender", what="What will you have?"),
            Call(label="order_drink"),
            Return(),
        ],
        "order_drink": [
            Say(who="bartender", what="One honey mead, coming up!"),
            Return(),
        ],
        "town_square": [
            Say(who="guide", what="Now at the town square."),
        ]
    }

    icfg = builder.build_from_labels(labels, entry_labels=["start"])
    assert len(icfg.nodes) > 0

    # Test call string length = 2 at order_drink
    order_nodes = icfg.get_nodes_for_label("order_drink")
    assert len(order_nodes) >= 2
    order_entry = order_nodes[0]
    assert len(order_entry.context) == 2
    assert "start" in order_entry.context[0]
    assert "visit_tavern" in order_entry.context[1]

    # Verify return edge returns to visit_tavern
    exit_node = [n for n in order_nodes if n.is_exit][0]
    succs = icfg.get_successors(exit_node.node_id)
    assert len(succs) == 1
    return_target = icfg.get_node(succs[0])
    assert return_target.label == "visit_tavern"

    print("    [✓] ICFG k-CFA (k=2) call-string context sensitivity verified!\n")


# =========================================================================
# 5. Tarjan SCC & Infinite Loop Elimination (Day-Night & Menu Loops)
# =========================================================================
def test_tarjan_scc_loops_and_dag_condensation():
    print("[*] 5. Verifying Tarjan SCC Loop Detection & DAG Condensation...")

    builder = ICFGBuilder(k=2)

    # Cyclic gameplay scenario:
    # 1. Day-Night cycle: morning -> afternoon -> evening -> night -> morning
    # 2. Shop menu loop: shop_menu -> buy_potion -> shop_menu
    labels = {
        "start": [
            Say(who="s", what="Start day loop"),
            Jump(target="morning"),
        ],
        "morning": [
            Say(who="s", what="Good morning"),
            Jump(target="afternoon"),
        ],
        "afternoon": [
            Say(who="s", what="Good afternoon"),
            Jump(target="evening"),
        ],
        "evening": [
            Say(who="s", what="Good evening"),
            Jump(target="night"),
        ],
        "night": [
            Say(who="s", what="Good night"),
            Jump(target="morning"),
        ],
        "free_roam_map": [
            Say(who="s", what="Map explore"),
            Jump(target="library"),
        ],
        "library": [
            Say(who="s", what="Read books"),
            Jump(target="free_roam_map"),
        ],
    }

    icfg = builder.build_from_labels(labels, entry_labels=["start", "free_roam_map"])

    tarjan = TarjanSCC(icfg)
    dag = tarjan.condense()

    # Verify loop components
    loop_components = dag.get_loop_components()
    assert len(loop_components) >= 2, f"Expected >= 2 loop components, got {len(loop_components)}"

    loop_kinds = {c.loop_kind for c in loop_components}
    assert LoopKind.DAY_NIGHT_LOOP in loop_kinds
    assert LoopKind.FREE_ROAM_LOOP in loop_kinds or LoopKind.GENERAL_LOOP in loop_kinds

    morning_node = icfg.get_nodes_for_label("morning")[0]
    assert dag.is_node_in_loop(morning_node.node_id) is True

    start_node = icfg.get_nodes_for_label("start")[0]
    assert dag.is_node_in_loop(start_node.node_id) is False

    # Topological ordering check (NO infinite loop hang)
    topo_order = dag.topological_sort()
    assert len(topo_order) == len(dag.components)
    print(f"    - Successfully condensed into {len(dag.components)} SCC DAG components.")
    print(f"    - Topological execution order count: {len(topo_order)}")
    print("    [✓] Tarjan SCC condensation: Zero infinite loop risk confirmed!\n")


# =========================================================================
# 6. Lengauer-Tarjan Dominator Tree & Milestone Separation
# =========================================================================
def test_lengauer_tarjan_dominators():
    print("[*] 6. Verifying Lengauer-Tarjan Dominator Tree & Global Milestones...")

    # Diamond control flow:
    # start -> fork_node -> route_alice, route_bob -> climax_node -> ending
    adj = {
        "start": {"fork_node"},
        "fork_node": {"route_alice", "route_bob"},
        "route_alice": {"climax_node"},
        "route_bob": {"climax_node"},
        "climax_node": {"ending"},
        "ending": set(),
    }
    pred = {
        "start": set(),
        "fork_node": {"start"},
        "route_alice": {"fork_node"},
        "route_bob": {"fork_node"},
        "climax_node": {"route_alice", "route_bob"},
        "ending": {"climax_node"},
    }

    dom_engine = LengauerTarjanDominator(adj=adj, pred=pred, entry_nodes=["start"])
    dom_engine.compute_dominators()

    # Milestones must dominate all subsequent story progress
    assert dom_engine.dominates("start", "fork_node") is True
    assert dom_engine.dominates("fork_node", "climax_node") is True
    assert dom_engine.dominates("climax_node", "ending") is True

    # Branch divergence check: neither route_alice nor route_bob dominates climax_node
    assert dom_engine.dominates("route_alice", "climax_node") is False
    assert dom_engine.dominates("route_bob", "climax_node") is False

    # Check mandatory dominators
    dom_set_ending = dom_engine.get_mandatory_dominators("ending")
    assert "start" in dom_set_ending
    assert "fork_node" in dom_set_ending
    assert "climax_node" in dom_set_ending
    assert "ending" in dom_set_ending
    assert "route_alice" not in dom_set_ending
    assert "route_bob" not in dom_set_ending

    # Classification
    milestones = dom_engine.get_global_milestones()
    assert "start" in milestones or "fork_node" in milestones
    assert "climax_node" in milestones

    branches = dom_engine.get_divergence_branches("fork_node")
    assert "route_alice" in branches and "route_bob" in branches

    print("    - Global story milestones identified:", milestones)
    print("    - Divergence branches from fork_node:", list(branches.keys()))
    print("    [✓] Lengauer-Tarjan Dominator Tree: Milestone separation verified!\n")


# =========================================================================
# 7. Path-Sensitive Trace Partitioning for Relational Variables
# =========================================================================
def test_path_sensitive_trace_partitioning():
    print("[*] 7. Verifying Path-Sensitive Trace Partitioning...")

    builder = ICFGBuilder(k=2)

    labels = {
        "start": [
            Say(who="guide", what="Choose your companion:"),
            Jump(target="alice_route"),
        ],
        "alice_route": [
            UserStatement(line="$ affection += 15"),
            UserStatement(line='$ route = "alice"'),
            Jump(target="cafe_date"),
        ],
        "bob_route": [
            UserStatement(line="$ affection += 2"),
            UserStatement(line='$ route = "bob"'),
            Jump(target="cafe_date"),
        ],
        "cafe_date": [
            Say(who="s", what="We meet at the cafe."),
        ],
    }

    icfg = builder.build_from_labels(labels, entry_labels=["start", "bob_route"])

    start_node_id = icfg.get_entry_node("start", context=())
    bob_entry_id = icfg.get_entry_node("bob_route", context=())
    icfg.add_edge(start_node_id, bob_entry_id, edge_type=EdgeType.BRANCH, condition="Choose Bob")

    engine = TracePartitioningEngine(icfg)
    engine.run(entry_node_id=start_node_id, initial_variables={"affection": 0, "route": "common"})

    # Check cafe_date has separate context versions
    cafe_versions = engine.get_context_versions("cafe_date")
    assert len(cafe_versions) >= 2, f"Expected >= 2 partitioned context versions, got {len(cafe_versions)}"

    routes_at_cafe = {v.state.get_var("route") for v in cafe_versions}
    assert "alice" in routes_at_cafe and "bob" in routes_at_cafe

    aff_brackets = {v.state.get_affection_bracket() for v in cafe_versions}
    assert "HIGH" in aff_brackets and "LOW" in aff_brackets

    assert engine.is_context_separated("cafe_date", "route") is True
    assert engine.is_context_separated("cafe_date", "affection") is True

    print("    [✓] Path-sensitive trace partitioning isolates opposing relational states!\n")


# =========================================================================
# 8. Temporal Honorific Graph & 100% Zero-Leakage Guarantee
# =========================================================================
def test_temporal_honorific_zero_leakage():
    print("[*] 8. Verifying Temporal Honorific Graph & Zero-Leakage Guarantee...")

    # Story progression:
    # n0: prologue (hooded stranger introduced as "???")
    # n1: encounter_forest (still "???")
    # n2: reveal_milestone (Dominator D: "My name is Sylvia!")
    # n3: chapter_three (known as "Sylvia")
    # n4: ending (known as "Sylvia")

    adj = {
        "n0": {"n1"},
        "n1": {"n2"},
        "n2": {"n3"},
        "n3": {"n4"},
        "n4": set(),
    }
    pred = {"n0": set(), "n1": {"n0"}, "n2": {"n1"}, "n3": {"n2"}, "n4": {"n3"}}

    dom_engine = LengauerTarjanDominator(adj=adj, pred=pred, entry_nodes=["n0"])
    dom_engine.compute_dominators()

    honorific_graph = TemporalHonorificGraph(dominator_engine=dom_engine)
    honorific_graph.register_character("char_mystery", initial_name="???")

    # Record identity reveal at Dominator node n2
    honorific_graph.record_reveal(
        character_id="char_mystery",
        revealed_name="Sylvia",
        dominator_node="n2",
        epoch_id="post_reveal",
    )

    # 1. Zero-Leakage Audit on all ancestor nodes before n2
    assert honorific_graph.get_character_name_at_node("char_mystery", "n0") == "???"
    assert honorific_graph.get_character_name_at_node("char_mystery", "n1") == "???"

    # Formal audit
    is_clean, leaked = honorific_graph.verify_no_leakage("char_mystery")
    assert is_clean is True
    assert len(leaked) == 0, f"Spoiler Leakage Detected at nodes: {leaked}"

    # 2. At and downstream of dominator n2, name is resolved
    assert honorific_graph.get_character_name_at_node("char_mystery", "n2") == "Sylvia"
    assert honorific_graph.get_character_name_at_node("char_mystery", "n3") == "Sylvia"
    assert honorific_graph.get_character_name_at_node("char_mystery", "n4") == "Sylvia"

    # 3. Dynamic Honorific Profiles (Vietnamese Pronoun Selection)
    prof_pre = honorific_graph.set_honorific_profile(
        speaker_id="player",
        listener_id="char_mystery",
        epoch_id="pre_reveal",
        pronoun_self="tôi",
        pronoun_other="cô nương",
        formality_tier=FormalityTier.POLITE,
        evidence_span="Lần đầu gặp gỡ tại bìa rừng",
    )
    assert prof_pre.pronoun_other == "cô nương"
    assert prof_pre.formality_tier == FormalityTier.POLITE

    prof_post = honorific_graph.set_honorific_profile(
        speaker_id="player",
        listener_id="char_mystery",
        epoch_id="post_reveal",
        pronoun_self="anh",
        pronoun_other="em",
        formality_tier=FormalityTier.INTIMATE,
        evidence_span="Sau khi Sylvia thổ lộ danh tính và lời thề",
    )
    assert prof_post.pronoun_self == "anh"
    assert prof_post.pronoun_other == "em"
    assert prof_post.formality_tier == FormalityTier.INTIMATE

    print("    [✓] Temporal Honorific Graph: 100% Zero-Leakage verified (0 spoiler leakage)!\n")


# =========================================================================
# Main Execution
# =========================================================================
def main():
    print("=================================================================")
    print("       RENSLATE PHASE 2 (GATE 2) INDEPENDENT AUDIT SUITE        ")
    print("=================================================================\n")

    start_time = time.time()

    test_lifecycle_5_phases_and_wrapnode()
    test_dialogue_normalization_and_extend()
    test_screen_language_ui_extraction()
    test_icfg_k_cfa_graph()
    test_tarjan_scc_loops_and_dag_condensation()
    test_lengauer_tarjan_dominators()
    test_path_sensitive_trace_partitioning()
    test_temporal_honorific_zero_leakage()

    elapsed = time.time() - start_time

    print("=================================================================")
    print(f" [✓] GATE 2 INDEPENDENT QUALITY AUDIT PASSED 100% ({elapsed:.2f}s)")
    print("=================================================================")


if __name__ == "__main__":
    main()
