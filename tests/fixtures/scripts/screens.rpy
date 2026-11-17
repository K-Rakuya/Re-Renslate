# Screen Language 2 (SL2) Fixture
screen main_menu():
    tag menu
    style_prefix "main_menu"

    vbox:
        text _("Renslate Test Game") size 36
        textbutton _("Start") action Start() tooltip _("Begin a new adventure")
        textbutton _("Load") action ShowMenu("load") tooltip _("Load saved game state")
        textbutton _("Preferences") action ShowMenu("preferences")
        textbutton _("Quit") action Quit(confirm=False)

screen say(who, what):
    style_prefix "say"

    window:
        id "window"
        if who is not None:
            text who id "who"
        text what id "what"
