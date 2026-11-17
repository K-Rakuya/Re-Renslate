# Ren'Py 7.x Sample Script (Legacy Python 2 compatible syntax)
init -1 python:
    score = 0
    persistent.game_beaten = False

define s = Character('Sylvia', color="#c8ffc8")
define m = Character('Narrator')

default player_name = "Traveler"

label start:
    $ score = 0
    s "Welcome to the old world!"
    extend " Are you prepared for the journey?"

    "The wind blows through the ancient ruins."

    menu:
        "Proceed forward":
            jump forward_path
        "Turn back":
            jump retreat_path

label forward_path:
    $ renpy.say(s, "Courage is rewarded.")
    s "Let us move forward, [player_name]!"
    return

label retreat_path:
    s "Perhaps another day."
    return
