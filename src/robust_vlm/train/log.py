import logging
import sys

logger_initialized = False

def setup_logger(log_file):
    global logger_initialized
    if logger_initialized:
        return
    logger_initialized = True

    level = logging.INFO
    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    if log_file is not None:
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        root.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)
