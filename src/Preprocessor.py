import logging
import misc_utils

from IOUtility import IOUtility
from OneHotEncoder import OneHotEncoder
from datetime import datetime


class Preprocessor:
    def __init__(self):
        self.io_utils = IOUtility()
        self.settings = self.io_utils.get_settings()
        self.profile = self.io_utils.get_profile()
        self.enc = None

    def load_encoder(self):
        """
        Loads OneHotEncoder object if present in cache provided 'cache_encoder' is True in settings.
        If any of the above condition is false, it creates a new encoder object, trains it and caches it.
        """
        encoder_cache_file_name = "encoder_small.cache"
        if self.settings['cache']['cache_encoder']:
            self.enc = self.io_utils.load_from_cache(encoder_cache_file_name)
        if self.enc is None:
            logging.debug("Encoder is not cached or cache is disabled")
            self.enc = self.__train_encoder(encoder_cache_file_name)
        else:
            logging.debug("Successfully loaded encoder from cache")
        return self.enc

    def encode(self, x):
        return self.enc.transform(x)

    def __train_encoder(self, encoder_cache_file_name):
        """
        Trains and caches the encoder in mini batch mode
        :param encoder_cache_file_name:
        :return: trained encoder object
        """

        model = self.io_utils.get_model()
        _use_cols = misc_utils.get_cols_to_load(labels=False, load_int_cols=False, load_cat_cols=True,
                                                ignore_features=model['ignore_features'])
        train_chunks = self.io_utils.load_train_set(use_cols=_use_cols, converters=misc_utils.get_converters(_use_cols),
                                                    chunk_size=1500000)
        enc = OneHotEncoder()
        start = datetime.now()

        logging.debug("Training Encoder...")
        for i, chunk in enumerate(train_chunks):
            enc.partial_fit(chunk.as_matrix())
            logging.debug("Trained encoder with chunk %d\t%s" % (i+1, str(datetime.now() - start)))
        logging.debug("Finished training encoder in \t%s" % str(datetime.now() - start))

        self.io_utils.save_to_cache(enc, encoder_cache_file_name)
        return enc
