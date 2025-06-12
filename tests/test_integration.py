
packages_all = {'analysis', 'derive', 'experiment', 'intensities', 'main', 'utils'}

def test_imports():
    print("\n\nTesting imports...\n")
    try:
        import wilson_suite as wilson
        assert set([a for a in dir(wilson) if '_' not in a]) == packages_all

    except ModuleNotFoundError as error:
        print('ModuleNotFoundError - ', error)

