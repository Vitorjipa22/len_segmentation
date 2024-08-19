from __future__ import annotations

import datetime as dt
import logging
import pathlib as pl
import sys
from logging.handlers import TimedRotatingFileHandler


class Logger:
    """A utility class for setting up and configuring a logger with both console and file handlers.

    This class simplifies the process of creating and configuring a logger for logging messages to the console and
    a rotating log file.

    Args:
        log_name (str): The name of the logger.
        log_dir (str): The directory where log files will be stored.
    """

    def __init__(self, log_name='auditor', log_dir='logs'):
        """Initialize the Logger instance with the specified log name and directory.

        Args:
            log_name (str): The name of the logger.
            log_dir (str): The directory where log files will be stored.
        """
        self.logger = logging.getLogger(log_name)
        if not self.logger.hasHandlers():
            self.logger.setLevel(logging.DEBUG)

            log_format = logging.Formatter('%(asctime)s %(levelname)s [%(module)s.%(funcName)s:%(lineno)d] %(message)s')

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(log_format)
            self.logger.addHandler(console_handler)

            log_dir = pl.Path(log_dir).expanduser().resolve()
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = TimedRotatingFileHandler(
                log_dir / f'{log_name}.log',
                when='midnight',
                atTime=dt.time(hour=1),
                utc=True,
            )
            file_handler.setFormatter(log_format)
            self.logger.addHandler(file_handler)

    def set_log_level(self, level):
        """Set the log level for the logger.

        Args:
            level (int): The log level (e.g., logging.INFO, logging.DEBUG, etc.).
        """
        self.logger.setLevel(level)

    def debug(self, message):
        """Log message as debug."""
        self.logger.debug(message)

    def info(self, message):
        """Log message as info."""
        self.logger.info(message)

    def warning(self, message):
        """Log message as warning."""
        self.logger.warning(message)

    def error(self, message):
        """Log message as error."""
        self.logger.error(message)

    def exception(self, message):
        """Log message as exception."""
        self.logger.exception(message)

    def critical(self, message):
        """Log message as critical."""
        self.logger.critical(message)


# Create a global logger instance
logger = Logger().logger

# Example usage:
if __name__ == '__main__':
    log = Logger()
    log.info('This is an info message.')
    log.warning('This is a warning message.')
    log.error('This is an error message.')
    log.exception('This is an exception message.')
    log.critical('This is a critical message.')
