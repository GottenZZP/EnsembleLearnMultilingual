import logging
import os


class Log:
    def __init__(self, log_name, log_type):
        self.logger = logging.getLogger('roberta_t')
        self.logger.setLevel(logging.DEBUG)

        path = os.path.join('../logging', log_type)

        self.consoleHandler = logging.StreamHandler()
        self.fileHandler = logging.FileHandler(os.path.join(path, log_name))

        self.fileHandler.setLevel(logging.INFO)

        _formatter = logging.Formatter("%(asctime)s|%(levelname)-8s|%(filename)10s%(lineno)4s|%(message)s")

        self.consoleHandler.setFormatter(_formatter)
        self.fileHandler.setFormatter(_formatter)

        self.logger.addHandler(self.consoleHandler)
        self.logger.addHandler(self.fileHandler)

    def writer_log(self, text, level='INFO'):
        if level == 'DEBUG':
            self.logger.debug(text)
        elif level == 'INFO':
            self.logger.info(text)
        elif level == 'WARNING':
            self.logger.warning(text)
        elif level == 'ERROR':
            self.logger.error(text)
        elif level == 'CRITICAL':
            self.logger.critical(text)
        else:
            self.logger.exception(text)
