import logging
import sys
from typing import Optional


def setup_logger(name: str = "network_ml", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a production-ready logging configuration with console and optional file logging.
    
    Args:
        name: Name of the logger.
        log_file: Optional absolute file path to store log histories.
        level: Standard logging levels (INFO, DEBUG, etc.).
        
    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if logger is re-initialized
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)

    # Unified professional format matching server output styles
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler for production audit trails
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
