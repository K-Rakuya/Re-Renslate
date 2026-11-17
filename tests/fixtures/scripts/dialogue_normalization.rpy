# Dialogue Normalization Test Script (4 Say forms & dynamic identity)
define a = Character("Alice")
define anon = DynamicCharacter("curr_name")
default curr_name = "???"

label dialogue_test:
    # 1. Explicit who != None
    a "Form 1: Explicit character dialogue."

    # 2. Narration who == None
    "Form 2: Implicit narration dialogue."

    # 3. Extend
    extend " Continued narration text."

    # 4. Direct renpy.say invocation
    $ renpy.say(a, "Form 4: Direct renpy.say call.")

    # UI localization strings
    $ msg1 = _("Save Game Now")
    $ msg2 = __("Deferred String Resolution")

    return
