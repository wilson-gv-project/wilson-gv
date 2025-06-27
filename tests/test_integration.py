
packages_all = {'analysis', 'derive', 'experiment', 'intensities', 'main', 'utils'}

def test_imports():
    print("\n\nTesting imports...\n")
    try:
        import wilson_suite as ws
        assert set([a for a in dir(ws) if '_' not in a]) == packages_all
        fail = False
        assert 'derive' in dir(ws)
        assert 'experiment' in dir(ws)
        assert 'intensities' in dir(ws)

    except ModuleNotFoundError as error:
        print('ModuleNotFoundError - ', error)
        fail = True
    assert not fail

def test_import_subpackages():
    print("\n\nTesting import of subpackages...\n")
    try:
        import wilson_suite as ws
        assert 'abstractions' in dir(ws.experiment)
        assert 'abstractions' in dir(ws.main)
        assert 'spectrum' in dir(ws.intensities)
        assert set([a for a in dir(ws) if '_' not in a]) == packages_all
        fail = False

    except ModuleNotFoundError as error:
        print('ModuleNotFoundError - ', error)
        fail = True
        assert not fail