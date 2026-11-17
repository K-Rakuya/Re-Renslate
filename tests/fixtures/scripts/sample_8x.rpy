# Ren'Py 8.x Modern Script (Python 3.12, SDK 8.4 - 8.5.3)
python early:
    EARLY_COMPILER_FLAG = True

init python:
    from renpy.revertable import RevertableDict, RevertableList

define e = Character(_("Eileen"), color="#4db2ff")
define mystery = DynamicCharacter("mystery_name", color="#ff77aa")

default player_name = "Player"
default mystery_name = "???"
default inventory = ["compass", "journal"]
default affection = 0

label start:
    mystery "You do not yet know who I am."
    "The protagonist observes the hooded figure."
    extend " There is a quiet stillness in the air."

    e "Hello, [player_name]! Welcome to Ren'Py 8.5.3."
    
    $ mystery_name = "Sylvia"
    mystery "At last, my true name is revealed."

    $ renpy.say(e, _("The journey begins now."))
    return
