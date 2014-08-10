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
    # str_to_int = lambda x: int(x) if len(x) > 0 else -100
    converters = dict()
    for c in load_cols:
        if c[0] == 'C':
            converters[c] = hex_to_int
        elif c != "Id" and c[0] == 'I':
            converters[c] = get_mean_normalizer_with_imputer(c)
    return converters


def remove_ignored_features(all_features, ignored_features=None):
    if ignored_features is not None:
        for feature in ignored_features:
            if feature in all_features:
                all_features.remove(feature)
    return all_features

def get_mean_normalizer_with_imputer(column_name):
    """
    Returns a function which imputes the missing value with mean and does a mean normalization
    
    The lambda function does a mean normalization. The formula for doing mean normalization of 
    value x from features Y is 
    (x - mean(Y))/range(Y).

    When missing value is imputed with the mean, on performing mean normalization, the value 
    becomes 0. So, the below lambdas returns 0 for NAs.
    """
    converter_funcs = {
        "I1": lambda x: (float(x) - 4) / 5775 if len(x) > 0 else 0,
        "I2": lambda x: (float(x) - 105.8) / 257678 if len(x) > 0 else 0,
        "I3": lambda x: (float(x) - 27) / 65535 if len(x) > 0 else 0,
        "I4": lambda x: (float(x) - 7) / 969 if len(x) > 0 else 0,
        "I5": lambda x: (float(x) - 18539) / 23159456 if len(x) > 0 else 0,
        "I6": lambda x: (float(x) - 116) / 431037 if len(x) > 0 else 0,
        "I7": lambda x: (float(x) - 16.3) / 56311 if len(x) > 0 else 0,
        "I8": lambda x: (float(x) - 12.52) / 6047 if len(x) > 0 else 0,
        "I9": lambda x: (float(x) - 106.1) / 29019 if len(x) > 0 else 0,
        "I10": lambda x: (float(x) - 1) / 11 if len(x) > 0 else 0,
        "I11": lambda x: (float(x) - 2.7) / 231 if len(x) > 0 else 0,
        # "I12": lambda x: (float(x) - 1) / 4008 if len(x) > 0 else 0,
        "I12": lambda x: -1 if len(x) == 0 else 0 if float(x) == 0.0 else 1,
        "I13": lambda x: (float(x) - 8) / 7393 if len(x) > 0 else 0
    }
    return converter_funcs[column_name]

