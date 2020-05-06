import logging
import sys
import typing

LOG_FILE = "/home/admin/logs/elasticdl.log"

_DEFAULT_LOGGER = "elasticdl.logger"
_STDOUT_LOGGER = "elasticdl_stdout.logger"

_DEFAULT_FORMATTER = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] "
    "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
)

_STREAM_HANDLER = logging.StreamHandler(stream=sys.stderr)
_STREAM_HANDLER.setFormatter(_DEFAULT_FORMATTER)


_LOGGER_CACHE = {}  # type: typing.Dict[str, logging.Logger]


def get_file_hander(filename):
    file_handler = logging.FileHandler(
        filename, mode='a', encoding="UTF-8", delay=True
    )
    file_handler.setFormatter(_DEFAULT_FORMATTER)
    return file_handler


def get_logger(
    name, level="INFO", filename=None, handlers=None, update=False
):
    if name in _LOGGER_CACHE and not update:
        return _LOGGER_CACHE[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = handlers or [_STREAM_HANDLER]
    logger.propagate = False

    if filename:
        file_handler = get_file_hander(filename)
        logger.handlers.append(file_handler)

    _LOGGER_CACHE[name] = logger
    return logger


stdout_logger = get_logger(_STDOUT_LOGGER)

default_logger = get_logger(_DEFAULT_LOGGER, filename=LOG_FILE)
