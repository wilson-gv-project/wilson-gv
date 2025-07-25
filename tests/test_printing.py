from wilson_utils.printing import (
    use_print_target,
    infoprint,
    debugprint,
    printtest,
    separatorprint,
    coolprint,
)
from wilson_utils.paths import UTILS_ROOT
TESTFILESDIR = UTILS_ROOT + "/tests/"

def test_debugfunc():
    pass

def test_debug_deep():
    pass

def test_set_print_target():
    separatorprint()
    import wilson_utils.printing as wprint
    import sys
    assert wprint.PRINT_TARGET == sys.stdout
    
    with use_print_target(open(TESTFILESDIR+"logfile.txt", "w")):
        infoprint('hello world')
    infoprint('hello again')

def test_infoprint():
    pass

def test_debugprint():
    pass

def test_printtest():
    pass

def test_separatorprint():
    pass

def test_coolprint():
    separatorprint()
    coolprint("This is a cool print test")
    coolprint("This is a cool print test", colorinfo='blue')
    
    # using `with` to not deal with opening and closing files
    with open(TESTFILESDIR + "coolprint_test.txt", "w") as f:
        coolprint("This is a cool print test", file=f)

    with open(TESTFILESDIR + "coolprint_test_red.txt", "w") as f:
        coolprint("This is a cool print test", colorinfo='red', file=f)

    pass