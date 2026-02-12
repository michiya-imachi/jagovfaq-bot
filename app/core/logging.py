import logging


_LOG_FORMAT = "[%(levelname)s] %(name)s: %(message)s"


def setup_logging(log_level: str) -> None:
    level = getattr(logging, str(log_level).upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    root_logger = logging.getLogger()
    formatter = logging.Formatter(_LOG_FORMAT)

    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)
        if handler.formatter is None:
            handler.setFormatter(formatter)
