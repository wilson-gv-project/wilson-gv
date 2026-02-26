from .testutils import MakeObjects

def test_feat_boxX():
    print()
    f = MakeObjects.mk_feature_single()
    print(f)
    print('\n-----------\n')
    f = MakeObjects.mk_features_non_ovrl()
    print(f)

    print('\n-----------\n')
    f = MakeObjects.mk_features_ovrl()
    print(f)