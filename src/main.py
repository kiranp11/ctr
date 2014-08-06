import logging

from IOUtility import IOUtility
from Preprocessor import Preprocessor
from Classifier import Classifier


class Main:
    def __init__(self):
        self.io_utils = IOUtility()
        self.profile = self.io_utils.get_profile()
        self.__configure_logging()

    def __configure_logging(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
         
        # create console handler and set level to info
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s", datefmt='%m/%d/%y %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # create debug file handler and set level to debug
        handler = logging.FileHandler(self.profile['log_file_path'],"a")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s", datefmt='%m/%d/%y %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def run(self):
        preprocessor = Preprocessor()
        preprocessor.load_encoder()

        classifier = Classifier(preprocessor)
        classifier.run()


def main():
    Main().run()


if __name__ == '__main__':
    main()