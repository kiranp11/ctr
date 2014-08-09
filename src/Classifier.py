import logging
import misc_utils
import scipy as sp

from IOUtility import IOUtility
from Preprocessor import Preprocessor
from scipy import sparse
from sklearn import linear_model, naive_bayes
from datetime import datetime


SEED = 42


class Classifier:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.io_utils = IOUtility()
        self.settings = self.io_utils.get_settings()
        self.models = {
                "SGDC": linear_model.SGDClassifier,
                "BNB": naive_bayes.BernoulliNB,
                "MNB": naive_bayes.MultinomialNB}

    def run(self):
        model, model_name, model_settings = self.__load_model()
        self.train(model, model_settings['ignore_features'])
        log_loss_score = self.predict(model, model_settings['ignore_features'])
        logging.info("Log loss score for model '%s' is '%f'" % (model_name, log_loss_score))

    def train(self, model, ignore_features):
        _use_cols = misc_utils.get_cols_to_load(labels=True, load_int_cols=True, load_cat_cols=True,
                                                ignore_features=ignore_features)
        train_chunks = self.io_utils.load_train_set(use_cols=_use_cols, converters=misc_utils.get_converters(_use_cols),
                                                    chunk_size=self.settings['chunk_size'])

        start = datetime.now()
        logging.debug("Training Model...")
        for i, chunk in enumerate(train_chunks):
            logging.debug("Processing chunk %d\t%s" % (i + 1, str(datetime.now() - start)))
            self.__process_train_chunk(chunk, model, ignore_features)
        logging.debug("Finished training in \t%s" % str(datetime.now() - start))

    def predict(self, model, ignore_features):
        _use_cols = misc_utils.get_cols_to_load(labels=True, load_int_cols=True, load_cat_cols=True,
                                                ignore_features=ignore_features)
        test_chunks = self.io_utils.load_validation_set(use_cols=_use_cols,
                                                        converters=misc_utils.get_converters(_use_cols),
                                                        chunk_size=self.settings['chunk_size'])
        start = datetime.now()
        logging.debug("Predicting probabilities...")
        actual_labels = []
        predicted_labels = []
        for i, chunk in enumerate(test_chunks):
            logging.debug("Processing chunk %d\t%s" % (i + 1, str(datetime.now() - start)))
            y, predictions = self.__process_test_chunk(chunk, model, ignore_features)
            actual_labels += y.tolist()
            predicted_labels += self.__cap_predictions(predictions)
        logging.debug("Finished predicting in \t%s" % str(datetime.now() - start))
        return self.evaluate_model(actual_labels, predicted_labels)

    def __process_test_chunk(self, chunk, model, ignore_features):
        y = chunk['Label']
        X = self.__preprocess(chunk, ignore_features)
        predictions = model.predict_proba(X)
        return y, predictions

    def __process_train_chunk(self, chunk, model, ignore_features):
        y = chunk['Label']
        X = self.__preprocess(chunk, ignore_features)
        model.partial_fit(X, y, classes=[0, 1])

    def __preprocess(self, chunk, ignore_features):
        int_features = sparse.coo_matrix(chunk[misc_utils.get_integer_cols(ignore_features=ignore_features)]).tocsr()
        cat_features = chunk[misc_utils.get_categorical_cols(ignore_features=ignore_features)].as_matrix()
        cat_features = self.preprocessor.encode(cat_features)
        X = sparse.hstack((int_features, cat_features))
        return X

    def __load_model(self):
        model_name = self.settings['model']
        model_id = model_name.split('_')[0]
        model = self.models[model_id](random_state=SEED)

        model_settings = self.io_utils.get_model(model_name)
        model_params = model_settings["params"]
        model.set_params(**model_params)
        return model, model_name, model_settings

    @staticmethod
    def evaluate_model(actual_labels, predicted_labels):
        def log_loss(act, pred):
            epsilon = 1e-15
            pred = sp.maximum(epsilon, pred)
            pred = sp.minimum(1-epsilon, pred)
            ll = sum(act*sp.log(pred) + sp.subtract(1, act)*sp.log(sp.subtract(1, pred)))
            ll = ll * -1.0/len(act)
            return ll

        return log_loss(actual_labels, predicted_labels)

    @staticmethod
    def __cap_predictions(preds):
        return_val = []
        for p in preds:
            prob = p[1]
            if prob > 0.98:
                prob = 0.98
            elif prob < 0.02:
                prob = 0.02
            return_val.append(prob)
        return return_val