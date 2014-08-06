import pandas as pd
import yaml
import cPickle as pickle
import os.path
import logging


class IOUtility:
    """
    Utility class to do file IO
    """

    def __init__(self):
        self.settings = yaml.load(open("../settings.yml"))
        self.models = yaml.load(open("../models.json"))
        self.profile = self.settings['profiles'][self.settings['profile']]

    def get_model(self, model=None):
        model = self.settings['model'] if model is None else model
        return self.models[model]

    def get_profile(self):
        return self.profile

    def get_settings(self):
        return self.settings

    def save_to_cache(self, obj, file_name):
        logging.debug("Caching '%s' object to '%s'" % (obj.__class__.__name__, file_name))
        pickle.dump(obj, open(self.profile['cache_location'] + file_name, 'w'))
        return

    def load_from_cache(self, file_name):
        cache_file_name = self.profile['cache_location'] + file_name
        logging.debug("Trying to load cache: '%s'" % cache_file_name)
        return None if not os.path.isfile(cache_file_name) else pickle.load(open(cache_file_name))

    def load_train_set(self, use_cols=None, converters=None, chunk_size=None):
        return self.__load_dataframe(self.profile['train_set_path'], use_cols, converters, chunk_size)

    def load_validation_set(self, use_cols=None, converters=None, chunk_size=None):
        return self.__load_dataframe(self.profile['validation_set_path'], use_cols, converters, chunk_size)

    def load_test_set(self, use_cols=None, converters=None, chunk_size=None):
        return self.__load_dataframe(self.profile['test_set_path'], use_cols, converters, chunk_size)

    @staticmethod
    def __load_dataframe(file_name, use_cols=None, converters=None, chunk_size=None):
        return pd.read_csv(file_name, usecols=use_cols, converters=converters, chunksize=chunk_size)