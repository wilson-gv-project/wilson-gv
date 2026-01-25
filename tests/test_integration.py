
packages_all = ['analysis', 'derive', 'experiment', 'intensities', 'main', 'utils']

def test_imports():
    print("\n\nTesting imports...\n")
    try:
        import wilson_suite as ws
        fail = False
        for i in packages_all:
            assert i in dir(ws)

    except ModuleNotFoundError as error:
        print('ModuleNotFoundError - ', error)
        fail = True
    assert not fail

def test_import_subpackages():
    print("\n\nTesting import of subpackages...\n")
    try:
        import wilson_suite as ws
        assert 'experiment_abstractions' in dir(ws.experiment)
        assert 'abstractions' in dir(ws.main)
        fail = False

    except ModuleNotFoundError as error:
        print('ModuleNotFoundError - ', error)
        fail = True
        assert not fail