"""
tests/test_graph_algorithms.py
Unit tests for static graph algorithms in renloc_core.static_graph:
  1. icfg.py: k-CFA (k=2 call-string context sensitivity), jump/call/return/expression.
  2. tarjan_scc.py: Tarjan SCC loop detection (day/night, free-roam, menu loops) & DAG condensation.
  3. dominator.py: Lengauer-Tarjan Dominator Tree & global milestones vs branch divergences.
  4. trace_partitioning.py: Path-sensitive trace partitioning for relational variables.
  5. honorific_graph.py: Temporal Honorific Graph & 100% ancestor zero-leakage guarantee.
"""
import pytest
from typing import Dict, List, Any

from renloc_core.ast_reader.ast_nodes import (
    Call,
    Jump,
    Label,
    Return,
    Say,
    Python,
    UserStatement,
)
from renloc_core.static_graph import (
    EdgeType,
    CallContext,
    ICFGNode,
    ICFGEdge,
    ICFG,
    ICFGBuilder,
    LoopKind,
    SCCComponent,
    CondensationDAG,
    TarjanSCC,
    DominatorRole,
    DominatorInfo,
    LengauerTarjanDominator,
    PartitionKey,
    AbstractState,
    PartitionedTraceNode,
    TracePartitioningEngine,
    UNKNOWN_IDENTITY,
    FormalityTier,
    CharacterIdentity,
    HonorificProfile,
    TemporalHonorificGraph,
)


# =========================================================================
# 1. ICFG with k-CFA (k=2) Tests
# =========================================================================

class TestICFGkCFA:
    def test_direct_jump_and_k_cfa_call_stack(self):
        builder = ICFGBuilder(k=2)

        # Procedures:
        # start:
        #   Say("s", "Hello")
        #   Call("sub_one")
        #   Say("s", "Back in start")
        #   Jump("end_label")
        #
        # sub_one:
        #   Say("s", "In sub one")
        #   Call("sub_two")
        #   Return()
        #
        # sub_two:
        #   Say("s", "In sub two")
        #   Return()
        #
        # end_label:
        #   Say("s", "The end")

        labels = {
            "start": [
                Say(who="s", what="Hello"),
                Call(label="sub_one"),
                Say(who="s", what="Back in start"),
                Jump(target="end_label"),
            ],
            "sub_one": [
                Say(who="s", what="In sub one"),
                Call(label="sub_two"),
                Return(),
            ],
            "sub_two": [
                Say(who="s", what="In sub two"),
                Return(),
            ],
            "end_label": [
                Say(who="s", what="The end"),
            ],
        }

        icfg = builder.build_from_labels(labels, entry_labels=["start"])

        # Check nodes exist
        assert len(icfg.nodes) > 0
        all_labels = icfg.get_labels()
        assert "start" in all_labels
        assert "sub_one" in all_labels
        assert "sub_two" in all_labels
        assert "end_label" in all_labels

        # Context sensitivity check for sub_two:
        # Context should be a tuple of length 2: (call_site_start, call_site_sub_one)
        sub_two_nodes = icfg.get_nodes_for_label("sub_two")
        assert len(sub_two_nodes) >= 2  # entry and return
        entry_sub_two = sub_two_nodes[0]
        assert len(entry_sub_two.context) == 2
        assert "start" in entry_sub_two.context[0]
        assert "sub_one" in entry_sub_two.context[1]

        # Verify Return edge goes back to sub_one
        sub_two_ret = [n for n in sub_two_nodes if n.is_exit][0]
        succs_of_sub_two_ret = icfg.get_successors(sub_two_ret.node_id)
        assert len(succs_of_sub_two_ret) >= 1
        return_target_node = icfg.get_node(succs_of_sub_two_ret[0])
        assert return_target_node is not None
        assert return_target_node.label == "sub_one"

    def test_dynamic_jump_expression(self):
        builder = ICFGBuilder(k=2)

        labels = {
            "start": [
                Say(who="s", what="Choose your path"),
                Jump(target="route_var", expression=True),
            ],
            "route_a": [Say(who="s", what="Path A")],
            "route_b": [Say(who="s", what="Path B")],
        }

        # Provide candidate dynamic jump targets
        icfg = builder.build_from_labels(
            labels,
            entry_labels=["start"],
            dynamic_jump_targets={"route_var": ["route_a", "route_b"]},
        )

        start_jump_node = icfg.get_nodes_for_label("start")[1]
        succs = icfg.get_successors(start_jump_node.node_id)
        assert len(succs) == 2  # targets route_a and route_b

        succ_labels = {icfg.get_node(s).label for s in succs}
        assert succ_labels == {"route_a", "route_b"}


# =========================================================================
# 2. Tarjan SCC & Condensation DAG Tests
# =========================================================================

class TestTarjanSCC:
    def test_loop_detection_and_dag_condensation(self):
        builder = ICFGBuilder(k=2)

        # Day/Night loop: morning -> afternoon -> evening -> night -> morning
        # Plus an exit to story_end when a condition is met
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
                # Fork: loop back to morning or break out to story_end
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

        # Wire an extra exit from night to free_roam_map
        labels["night"] = [
            Say(who="s", what="Good night"),
            # In Ren'Py AST, branch edges can be modeled directly
            Jump(target="morning"),
        ]

        icfg = builder.build_from_labels(labels, entry_labels=["start", "free_roam_map"])

        tarjan = TarjanSCC(icfg)
        dag = tarjan.condense()

        # Verify loop components
        loop_comps = dag.get_loop_components()
        assert len(loop_comps) >= 2  # Day/night loop + Free roam map loop

        loop_kinds = {c.loop_kind for c in loop_comps}
        assert LoopKind.DAY_NIGHT_LOOP in loop_kinds
        assert LoopKind.FREE_ROAM_LOOP in loop_kinds

        # Verify node loop tagging
        morning_node = icfg.get_nodes_for_label("morning")[0]
        assert dag.is_node_in_loop(morning_node.node_id) is True

        start_node = icfg.get_nodes_for_label("start")[0]
        assert dag.is_node_in_loop(start_node.node_id) is False

        # Verify Condensation DAG is strictly acyclic via topological sort
        topo = dag.topological_sort()
        assert len(topo) == len(dag.components)


# =========================================================================
# 3. Lengauer-Tarjan Dominator Tree Tests
# =========================================================================

class TestLengauerTarjanDominator:
    def test_milestone_and_branch_divergence_separation(self):
        # Diamond control flow:
        # start (Milestone 1)
        #   -> fork_node (Milestone 2)
        #        /         \
        #  route_alice   route_bob (Branch divergences)
        #        \         /
        #   climax_node (Milestone 3)
        #        -> ending (Terminal)

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

        dom_engine = LengauerTarjanDominator(
            adj=adj,
            pred=pred,
            entry_nodes=["start"],
        )
        idoms = dom_engine.compute_dominators()

        # start dominates fork_node
        assert dom_engine.dominates("start", "fork_node")
        # fork_node dominates route_alice and route_bob
        assert dom_engine.dominates("fork_node", "route_alice")
        assert dom_engine.dominates("fork_node", "route_bob")

        # Crucial: route_alice does NOT dominate climax_node (because path through route_bob bypasses it)
        assert dom_engine.dominates("route_alice", "climax_node") is False
        assert dom_engine.dominates("route_bob", "climax_node") is False

        # fork_node DOES dominate climax_node (every path through diamond passes through fork_node)
        assert dom_engine.dominates("fork_node", "climax_node") is True
        # climax_node dominates ending
        assert dom_engine.dominates("climax_node", "ending") is True

        # Check mandatory dominator set for ending
        dom_set_ending = dom_engine.get_mandatory_dominators("ending")
        assert "start" in dom_set_ending
        assert "fork_node" in dom_set_ending
        assert "climax_node" in dom_set_ending
        assert "ending" in dom_set_ending
        assert "route_alice" not in dom_set_ending
        assert "route_bob" not in dom_set_ending

        # Check global milestones
        milestones = dom_engine.get_global_milestones()
        assert "start" in milestones or "fork_node" in milestones
        assert "climax_node" in milestones

        # Check branch divergence partitioning at fork_node
        branches = dom_engine.get_divergence_branches("fork_node")
        assert "route_alice" in branches
        assert "route_bob" in branches
        assert "route_alice" in branches["route_alice"]
        assert "route_bob" not in branches["route_alice"]


# =========================================================================
# 4. Path-Sensitive Trace Partitioning Tests
# =========================================================================

class TestTracePartitioning:
    def test_relational_state_tracking_and_context_separation(self):
        builder = ICFGBuilder(k=2)

        # Story:
        # start:
        #   choice:
        #     "Help Alice" -> jump to alice_route
        #     "Help Bob" -> jump to bob_route
        #
        # alice_route:
        #   $ affection += 15
        #   $ route = "alice"
        #   jump cafe_date
        #
        # bob_route:
        #   $ affection += 2
        #   $ route = "bob"
        #   jump cafe_date
        #
        # cafe_date:
        #   Say("s", "We meet at the cafe.")

        labels = {
            "start": [
                Say(who="s", what="Make your decision"),
                # Simulate menu choice branching with Jump
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

        # Build ICFG with both entries to explore parallel branches
        icfg = builder.build_from_labels(labels, entry_labels=["start", "bob_route"])

        # Also wire branch from start to bob_route
        start_node_id = icfg.get_entry_node("start", context=())
        bob_entry_id = icfg.get_entry_node("bob_route", context=())
        icfg.add_edge(start_node_id, bob_entry_id, edge_type=EdgeType.BRANCH, condition="Choose Bob")

        engine = TracePartitioningEngine(icfg)
        engine.run(entry_node_id=start_node_id, initial_variables={"affection": 0, "route": "common"})

        # Verify that cafe_date has separate context versions
        cafe_versions = engine.get_context_versions("cafe_date")
        assert len(cafe_versions) >= 2

        # Check route_id separation
        routes_at_cafe = {v.state.get_var("route") for v in cafe_versions}
        assert "alice" in routes_at_cafe
        assert "bob" in routes_at_cafe

        # Check affection brackets separation (Alice=HIGH, Bob=LOW)
        aff_brackets_at_cafe = {v.state.get_affection_bracket() for v in cafe_versions}
        assert "HIGH" in aff_brackets_at_cafe
        assert "LOW" in aff_brackets_at_cafe

        # Verify helper query is_context_separated
        assert engine.is_context_separated("cafe_date", "route") is True
        assert engine.is_context_separated("cafe_date", "affection") is True


# =========================================================================
# 5. Temporal Honorific Graph & Zero-Leakage Tests
# =========================================================================

class TestTemporalHonorificGraph:
    def test_anonymous_reveal_zero_leakage_and_honorific_profiles(self):
        # Build story control flow:
        # node_0: start (introduce '???')
        # node_1: prologue_interaction (still '???')
        # node_2: reveal_milestone (Dominator node D: "My name is Eileen!")
        # node_3: chapter_one (known as 'Eileen')
        # node_4: epilogue (known as 'Eileen')

        adj = {
            "node_0": {"node_1"},
            "node_1": {"node_2"},
            "node_2": {"node_3"},
            "node_3": {"node_4"},
            "node_4": set(),
        }
        pred = {
            "node_0": set(),
            "node_1": {"node_0"},
            "node_2": {"node_1"},
            "node_3": {"node_2"},
            "node_4": {"node_3"},
        }

        dom_engine = LengauerTarjanDominator(adj=adj, pred=pred, entry_nodes=["node_0"])
        dom_engine.compute_dominators()

        honorific_graph = TemporalHonorificGraph(dominator_engine=dom_engine)
        honorific_graph.register_character("c_eileen", initial_name="???")

        # Record reveal at dominator node 'node_2'
        honorific_graph.record_reveal(
            character_id="c_eileen",
            revealed_name="Eileen",
            dominator_node="node_2",
            epoch_id="post_reveal",
        )

        # 1. 100% Ancestors of node_2 MUST retain UNKNOWN_IDENTITY (Zero Leakage)
        assert honorific_graph.get_character_name_at_node("c_eileen", "node_0") == "???"
        assert honorific_graph.get_character_name_at_node("c_eileen", "node_1") == "???"

        # Formal leakage audit
        is_leakage_free, leaked = honorific_graph.verify_no_leakage("c_eileen")
        assert is_leakage_free is True
        assert len(leaked) == 0

        # 2. At and downstream of reveal milestone node_2, name is resolved
        assert honorific_graph.get_character_name_at_node("c_eileen", "node_2") == "Eileen"
        assert honorific_graph.get_character_name_at_node("c_eileen", "node_3") == "Eileen"
        assert honorific_graph.get_character_name_at_node("c_eileen", "node_4") == "Eileen"

        # 3. Temporal Honorific Profiles across epochs
        # Epoch 1: Pre-reveal stranger
        prof_pre = honorific_graph.set_honorific_profile(
            speaker_id="mc",
            listener_id="c_eileen",
            epoch_id="epoch_pre_reveal",
            pronoun_self="tôi",
            pronoun_other="cô gái lạ",
            formality_tier=FormalityTier.POLITE,
            evidence_span="Gặp gỡ lần đầu trong đêm mưa",
        )
        assert prof_pre.pronoun_other == "cô gái lạ"
        assert prof_pre.formality_tier == FormalityTier.POLITE

        # Epoch 2: Post-reveal romantic relationship
        prof_post = honorific_graph.set_honorific_profile(
            speaker_id="mc",
            listener_id="c_eileen",
            epoch_id="epoch_post_reveal",
            pronoun_self="anh",
            pronoun_other="em",
            formality_tier=FormalityTier.INTIMATE,
            evidence_span="Sau khi biết tên Eileen và cùng vượt qua thử thách",
        )
        assert prof_post.pronoun_self == "anh"
        assert prof_post.pronoun_other == "em"
        assert prof_post.formality_tier == FormalityTier.INTIMATE

        # Retrieve and verify
        fetched = honorific_graph.get_honorific_profile("mc", "c_eileen", "epoch_post_reveal")
        assert fetched is not None
        assert fetched.pronoun_self == "anh"
        assert fetched.pronoun_other == "em"
