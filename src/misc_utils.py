def get_cols_to_load(labels=True, load_int_cols=True, load_cat_cols=True, ignore_features=None):
    cols_to_load = []
    if labels:
        cols_to_load += ["Id", "Label"]
    if load_int_cols:
        cols_to_load += get_integer_cols()
    if load_cat_cols:
        cols_to_load += get_categorical_cols()

    cols_to_load = remove_ignored_features(cols_to_load, ignore_features)
    return cols_to_load


def get_categorical_cols(ignore_features=None):
    features = ['C' + str(i) for i in range(1, 27)]
    return remove_ignored_features(features, ignore_features)


def get_integer_cols(ignore_features=None):
    features = ['I' + str(i) for i in range(1, 14)]
    return remove_ignored_features(features, ignore_features)


def get_converters(load_cols):
    """
    Returns converters to convert categorical hash features to int
    :param load_cols:
    """
    hex_to_int = lambda x: int(x, 16) if len(x) > 0 else 0
    str_to_int = lambda x: int(x) if len(x) > 0 else -100
    converters = dict()
    for c in load_cols:
        if c[0] == 'C':
            converters[c] = hex_to_int
        elif c[0] == 'I':
            converters[c] = str_to_int
    return converters


def remove_ignored_features(all_features, ignored_features=None):
    if ignored_features is not None:
        for feature in ignored_features:
            if feature in all_features:
                all_features.remove(feature)
    return all_features