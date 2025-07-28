from wilson_utils.printing import (
    use_print_target,
    infoprint,
    debugprint,
    printtest,
    separatorprint,
    coolprint,
    debugfunc,
    debug_deep
)
import sys

from wilson_utils.paths import UTILS_ROOT
TESTFILESDIR = UTILS_ROOT + "/tests/"

def test_debugfunc(capsys):
    """
    capsys - pytest caturing stdout/stderr
    """
    import wilson_utils.printing as wprinting
    wprinting.level = 2
    wprinting.PRINT_TARGET = sys.stdout

    from sys import stdout
    wprinting._styled_print("DEBUG][debugfunc", "95", "hello test_debugfunc", file=stdout)

    captured = capsys.readouterr()
    assert "[DEBUG][debugfunc] hello test_debugfunc" in captured.out

    debugfunc('hello debugfunc', 'yo')
    captured = capsys.readouterr()
    assert "[DEBUG][yo] hello debugfunc" in captured.out


def test_debug_deep(capsys):
    """
    capsys - pytest caturing stdout/stderr
    """
    import wilson_utils.printing as wprinting
    wprinting.level = 2
    wprinting.PRINT_TARGET = sys.stdout

    from sys import stdout
    wprinting._styled_print("DEBUG DEEP][debug_deep", "95", "hello debug_deep", file=stdout)

    captured = capsys.readouterr()
    assert "[DEBUG DEEP][debug_deep] hello debug_deep" in captured.out

    debug_deep('hello debug_deep', 'yo')
    captured = capsys.readouterr()
    assert "[DEBUG DEEP][yo] hello debug_deep" in captured.out

def test_separatorprint(capsys):
    import wilson_utils.printing as wprint
    wprint.PRINT_TARGET = sys.stdout # reseting
    separatorprint()
    captured = capsys.readouterr()
    assert '---------------  ---------------' in captured.out

def test_set_print_target():
    import wilson_utils.printing as wprint
    # wprint.PRINT_TARGET = sys.stdout # reseting
    separatorprint()

    assert wprint.PRINT_TARGET == sys.stdout
    
    with use_print_target(open(TESTFILESDIR+"logfile0.txt", "w")):
        infoprint('hello world')
    infoprint('hello again')
    with open(TESTFILESDIR+"logfile0.txt", "r") as f:
        lines = f.readlines()
    assert lines[0].strip() == '[INFO] hello world'

def test_infoprint():
    # import wilson_utils.printing as wprint
    # wprint.PRINT_TARGET = sys.stdout # reseting
    separatorprint()
    
    with use_print_target(open(TESTFILESDIR+"logfile1.txt", "a")):
        infoprint('hello test_infoprint')
    infoprint('hello test_infoprint again')
    with open(TESTFILESDIR+"logfile1.txt", "r") as f:
        lines = f.readlines()
    assert lines[0].strip() == '[INFO] hello test_infoprint'

def test_debugprint():
    separatorprint()
    with use_print_target(open(TESTFILESDIR+"logfile2.txt", "a")):
        debugprint('hello test_debugprint')
    debugprint('hello test_debugprint again')
    with open(TESTFILESDIR+"logfile2.txt", "r") as f:
        lines = f.readlines()
    assert lines[0].strip() == '[DEBUG] hello test_debugprint'

def test_printtest():
    separatorprint()
    with use_print_target(open(TESTFILESDIR+"logfile3.txt", "a")):
        printtest('hello test_printtest')
    printtest('hello test_printtest again')
    with open(TESTFILESDIR+"logfile3.txt", "r") as f:
        lines = f.readlines()
    assert lines[0].strip() == '[TESTPRINT] hello test_printtest'

def test_coolprint():
    separatorprint()
    coolprint("This is a cool print test")
    coolprint("This is a cool print test", colorinfo='blue')
    
    # using `with` to not deal with opening and closing files
    with open(TESTFILESDIR + "coolprint_test.txt", "w") as f:
        coolprint("This is a cool print test", file=f)
    with open(TESTFILESDIR+"coolprint_test.txt", "r") as f:
        lines = f.readlines()
    assert lines[0].strip() == 'This is a cool print test'

    with open(TESTFILESDIR + "coolprint_test_red.txt", "w") as f:
        coolprint("This is a cool print test", colorinfo='red', file=f)
    with open(TESTFILESDIR+"coolprint_test_red.txt", "r") as f:
        lines = f.readlines()
    assert lines[0].strip() == 'This is a cool print test'

