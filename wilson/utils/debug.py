"""
Debug print functions


"""
from typing import Any

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
