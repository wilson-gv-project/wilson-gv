"""
Debug print functions


"""
level = 0

def debugfunc(msgs: str, tag: str = "") -> None:
    """
    debug print with tags at level >= 1
    """
    if level >= 1:
        # fill in on the right to given length of string, fill with spaces
        print(f"\033[95m[DEBUG][{tag}] \033[0m {msgs}".ljust(40, ' '))

def debug_deep(msgs: str, tag: str = "") -> None:
    """
    debug print with tags at level >= 2
    """
    if level >= 2:
        print(f"\033[95m[DEBUG DEEP][{tag}] \033[0m {msgs}".ljust(44, ' '))

def info_message(msgs: str):
    print(f"\033[93m[INFO] {msgs} \033[0m".ljust(44, ' '))


def separator_print(title: str):
    print('\n'+'-'*15+f' {title} '+'-'*15)

def colordebug(text):
    print(f'\033[92m[DEBUG] {text} \033[0m')