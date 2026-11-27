import pickle


def pickle_this_to(obj, filenamepkl: str, save_to: str = ''):
    import os
    filepath = os.path.abspath(os.path.join(save_to, filenamepkl))

    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def unpickle_smth_from(filenamepkl: str, load_from: str = ''):
    import os
    filepath = os.path.abspath(os.path.join(load_from, filenamepkl))

    with open(filepath, "rb") as f:
        loaded_obj = pickle.load(f)

    return loaded_obj
