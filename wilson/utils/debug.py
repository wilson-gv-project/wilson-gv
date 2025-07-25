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



def infoprint(*msgs: str):
    text = ''
    for m in msgs:
        if not isinstance(m, str):
            m = repr(m)
        text += m
    print(f"\033[93m[INFO] {msgs} \033[0m".ljust(44, ' '))

def separator_print(*title: str):
    if not isinstance(title, str):
        title = repr(title)
    print('\n\033[95m'+'-'*15+f' {title} '+'-'*15+'\033[0m')

def debugprint(*msgs):
    text = ''
    for m in msgs:
        if not isinstance(m, str):
            m = repr(m)
        text += m
    print(f'\033[92m[DEBUG] {text} \033[0m')

def printtest(*msgs: Any):
    text = ''
    for m in msgs:
        if not isinstance(m, str):
            m = repr(m)
        text += m
    print(f'\033[94m[TESTPRINT] {text} \033[0m')