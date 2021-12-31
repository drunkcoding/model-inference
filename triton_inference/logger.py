import logging
import sys
from logging import handlers

class Logger():

    def __init__(self, file, level, rollsize, backup):        
        if level.lower() == "debug":
            self.level = logging.DEBUG
        elif level.lower() == "info":
            self.level = logging.INFO
        else:
            raise ValueError('logging level must be INFO or DEBUG')

        self.logger = logging.getLogger()
        self.logger.setLevel(self.level)

        format = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(process)d [%(filename)s - %(lineno)d] -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
        )

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(format)
        self.logger.addHandler(ch)

        filename = file.split(".")[0]

        fh = handlers.RotatingFileHandler(f"{filename}.log", maxBytes=rollsize, backupCount=backup)
        fh.setFormatter(format)
        self.logger.addHandler(fh)

    def info(self, pattern, *args):
        self.logger.info(pattern, *args)

    def fatal(self, pattern, *args):
        self.logger.fatal(pattern, *args)

    def debug(self, pattern, *args):
        self.logger.debug(pattern, *args)

    def error(self, pattern, *args):
        self.logger.error(pattern, *args)

    def warn(self, pattern, *args):
        self.logger.warn(pattern, *args)
 
if __name__ == "__main__":
    logger = Logger(__file__, "info", 0, 0)
    logger.info("abcdefg")

