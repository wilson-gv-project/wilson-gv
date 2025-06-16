
enabled = False
level = 1

def debugfunc(msgs, tag=""):
    if level >= 1:
        # fill in on the right to given length of string, fill with spaces
        print(f"\033[95m[DEBUG][{tag}] \033[0m {msgs}".ljust(40, ' '))
    # if level >= 1:
    #     print(f"\033[95m[INFO][{tag}] \033[0m {msgs}".ljust(45, ' '))

def debug_deep(msgs, tag=""):
    if level >= 2:
        print(f"\033[95m[DEBUG DEEP][{tag}] \033[0m {msgs}".ljust(44, ' '))