"""
tests/test_static_graph_phase2.py
Comprehensive unit tests for static_graph components:
  1. lifecycle.py: Ren'Py 5-phase initialization ordering & WrapNode tracking.
  2. dialogue_normalizer.py: 4 dialogue forms normalization (who!=None, who==None, extend, renpy.say, _/__).
  3. screen_extractor.py: sl2.slast UI string extraction and LayeredImage emotional/costume state tracking.
"""
from pathlib import Path
import pytest

from renloc_core.ast_reader.ast_nodes import (
    EarlyPython,
    Init,
    Define,
    Default,
    Label,
    Say,
    TranslateSay,
    Python,
    UserStatement,
    Screen,
    SLScreen,
    SLDisplayable,
    SLIf,
    SLFor,
    SLPython,
    Show,
    Scene,
    LayeredImage,
    RawAttribute,
)
from renloc_core.static_graph import (
    LifecyclePhase,
    WrapNode,
    LifecycleSimulator,
    DialogueForm,
    DialogueEntity,
    DialogueNormalizer,
    ExtractedUIString,
    ScreenExtractor,
    LayeredImageTracker,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# =========================================================================
# 1. Lifecycle 5-Phase Simulator Tests
# =========================================================================

class TestLifecycleSimulator:
    def test_5_phase_ordering_and_wrapnode(self):
        simulator = LifecycleSimulator()

        # Build heterogeneous AST representing all 5 phases out-of-order
        raw_nodes = [
            Label(name="start", loc=("story.rpy", 50)),
            Say(who="s", what="Story begins", loc=("story.rpy", 51)),
            Default(varname="affection", code="0", loc=("vars.rpy", 10)),
            Init(priority=10, block=[Define(varname="late_char", code="Character()", loc=("late.rpy", 5))], loc=("late.rpy", 1)),
            Init(priority=-1, block=[Define(varname="early_char", code="Character()", loc=("early.rpy", 2))], loc=("early.rpy", 1)),
            Define(varname="s", code="Character('Sylvia')", loc=("defs.rpy", 20)), # priority 0
            EarlyPython(code="EARLY_FLAG = True", loc=("early.rpy", 1)),
            Label(name="main_menu", loc=("screens.rpy", 100)),
            Label(name="splashscreen", loc=("splash.rpy", 5)),
        ]

        simulator.wrap_ast(raw_nodes)
        ordered = simulator.get_execution_order()

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

    def test_simulate_state_environment(self):
        simulator = LifecycleSimulator()
        nodes = [
            EarlyPython(code="FLAG = 1", loc=("setup.rpy", 1)),
            Define(varname="hero", code="'Arthur'", loc=("defs.rpy", 5)),
            Default(varname="hp", code="100", loc=("stats.rpy", 10)),
        ]
        simulator.wrap_ast(nodes)
        state = simulator.simulate_state()

        assert "hero" in state["defined_characters"]
        assert "hp" in state["default_variables"]
        assert len(state["early_flags"]) == 1


# =========================================================================
# 2. Dialogue Normalizer Tests (4 Forms & Extend)
# =========================================================================

class TestDialogueNormalizer:
    def test_4_forms_normalization(self):
        normalizer = DialogueNormalizer()

        nodes = [
            # Form 1: Explicit who != None
            Say(who="alice", what="Form 1: Explicit character.", loc=("script.rpy", 10)),

            # Form 3: Extend statement inheriting Alice
            Say(who="extend", what=" Continued text with more details.", loc=("script.rpy", 11)),

            # Form 2: Narration who == None
            Say(who=None, what="Form 2: Narration text.", loc=("script.rpy", 15)),

            # Form 3: Extend statement inheriting Narration
            Say(who="extend", what=" Narration continues.", loc=("script.rpy", 16)),

            # Form 4a: Direct renpy.say() in Python
            UserStatement(line='$ renpy.say(alice, "Form 4a: Direct renpy.say call.")', loc=("script.rpy", 20)),

            # Form 4b & 4c: _(s) immediate and __(s) deferred in Python
            UserStatement(line='$ msg1 = _("Save Game Now")\n$ msg2 = __("Deferred String")', loc=("script.rpy", 25)),
        ]

        entities = normalizer.normalize_stream(nodes, current_label="test_scene")

        assert len(entities) == 7

        # 1. Explicit
        e1 = entities[0]
        assert e1.form == DialogueForm.EXPLICIT
        assert e1.who == "alice"
        assert e1.what == "Form 1: Explicit character."
        assert e1.dialogue_index == 1

        # 2. Extend Alice
        e2 = entities[1]
        assert e2.form == DialogueForm.EXTEND
        assert e2.who == "alice"  # Inherited from e1!
        assert e2.parent_id == e1.entity_id
        assert e2.dialogue_index == 2

        # 3. Narration
        e3 = entities[2]
        assert e3.form == DialogueForm.NARRATION
        assert e3.who is None
        assert e3.what == "Form 2: Narration text."

        # 4. Extend Narration
        e4 = entities[3]
        assert e4.form == DialogueForm.EXTEND
        assert e4.who is None  # Inherited None from e3!
        assert e4.parent_id == e3.entity_id

        # 5. Direct renpy.say
        e5 = entities[4]
        assert e5.form == DialogueForm.RENPY_SAY
        assert e5.who == "alice"
        assert "Direct renpy.say call" in e5.what

        # 6. _(s) Immediate
        e6 = entities[5]
        assert e6.form == DialogueForm.TRANSLATABLE_STRING
        assert e6.what == "Save Game Now"
        assert e6.is_deferred is False

        # 7. __(s) Deferred
        e7 = entities[6]
        assert e7.form == DialogueForm.TRANSLATABLE_STRING
        assert e7.what == "Deferred String"
        assert e7.is_deferred is True


# =========================================================================
# 3. Screen Extractor & LayeredImage Tracker Tests
# =========================================================================

class TestScreenExtractorAndLayeredImage:
    def test_screen_extractor_slast_tree(self):
        extractor = ScreenExtractor()

        # Build SLScreen AST tree with text, textbutton, tooltip, alt, if, for, python
        btn_disp = SLDisplayable(
            loc=("screens.rpy", 10),
            name="textbutton",
            positional=['_("Start Game")'],
            keyword=[("tooltip", '_("Begin new game")'), ("alt", '"Accessibility Alt Text"')],
        )
        text_disp = SLDisplayable(
            loc=("screens.rpy", 15),
            name="text",
            positional=['"Title Screen"'],
        )
        cond_branch = SLIf(
            loc=("screens.rpy", 20),
            entries=[
                ('_("Admin Mode")', [SLDisplayable(name="text", positional=['"Admin Active"'])]),
            ]
        )
        py_node = SLPython(
            loc=("screens.rpy", 25),
            code=Python(code='notify_text = _("Update Available")'),
        )

        screen_ast = SLScreen(
            loc=("screens.rpy", 1),
            name="main_menu",
            children=[btn_disp, text_disp, cond_branch, py_node],
        )

        extracted = extractor.extract_screen(screen_ast)
        texts = [e.text for e in extracted]
        types = [e.source_type for e in extracted]

        assert "Start Game" in texts
        assert "Begin new game" in texts
        assert "Accessibility Alt Text" in texts
        assert "Title Screen" in texts
        assert "Admin Mode" in texts
        assert "Admin Active" in texts
        assert "Update Available" in texts

        assert "textbutton" in types
        assert "tooltip" in types
        assert "alt" in types
        assert "condition" in types
        assert "python" in types

    def test_layeredimage_state_tracking_and_dialogue_tagging(self):
        tracker = LayeredImageTracker()

        # 1. Register LayeredImage
        li = LayeredImage(
            name="sylvia",
            attributes=[
                RawAttribute(attribute="happy"),
                RawAttribute(attribute="sad"),
                RawAttribute(attribute="school_uniform"),
            ]
        )
        tracker.register_layered_image(li)
        assert "happy" in tracker.image_definitions["sylvia"]

        # 2. Process Show statement: show sylvia happy school_uniform
        tracker.process_show_or_scene(Show(imspec=("sylvia", "happy", "school_uniform")))
        active_tags = tracker.get_active_tags("sylvia")
        assert active_tags == {"happy", "school_uniform"}

        # 3. Tag dialogue entity spoken by Sylvia
        dlg = DialogueEntity(
            entity_id="d1",
            who="sylvia",
            what="I am so glad to see you!",
            form=DialogueForm.EXPLICIT,
            filename="script.rpy",
            linenumber=10,
            block_depth=0,
            translation_id="t1",
        )
        tracker.tag_dialogue(dlg)
        assert "happy" in dlg.context_tags
        assert "school_uniform" in dlg.context_tags

        # 4. Scene resets layers
        tracker.process_show_or_scene(Scene(imspec=("bg", "ruins")))
        assert tracker.get_active_tags("sylvia") == set()
