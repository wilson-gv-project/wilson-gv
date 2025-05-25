
enabled = False
level = 2

def debugfunc(msgs, tag=""):
    if level >= 2:
        print(f"\033[95m[DEBUG][{tag}] \033[0m {msgs}".ljust(40, ' '))
    # if level >= 1:
    #     print(f"\033[95m[INFO][{tag}] \033[0m {msgs}".ljust(45, ' '))
