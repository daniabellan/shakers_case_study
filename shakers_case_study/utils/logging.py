import inspect
import logging
import logging.config
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import yaml


def setup_logging(config_filename: str = "logging_config.yaml", env: str = "development"):
    """
    Set up logging using a YAML configuration file.
    If the file is not found, falls back to a basic default configuration.

    Args:
        config_filename (str): Relative path to the YAML logging configuration file.
        env (str): Runtime environment. Affects the log level
        ('development' = DEBUG, 'production' = INFO).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = Path(__file__).resolve().parent.parent.parent  # shakers_case_study/
    config_path = base_dir / config_filename

    log_dir = base_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"{timestamp}.log"

    if config_path.exists():
        with config_path.open("r") as f:
            config = yaml.safe_load(f)

        # Adjust log level depending on the environment
        if env == "production":
            config["root"]["level"] = "INFO"
        elif env == "development":
            config["root"]["level"] = "DEBUG"

        # Set dynamic log filename
        config["handlers"]["file"]["filename"] = str(log_file)

        # Apply the logging configuration
        logging.config.dictConfig(config)
    else:
        # Fallback: basic configuration
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
        )


def setup_daily_logging(service_name: str = "chatbot"):
    """
    Set up daily rotating logs. Each log file is rotated at midnight, keeping up to 7 backups.

    Args:
        service_name (str): Base name for the log file (e.g., 'chatbot', 'ingestion').
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"{service_name}.log"

    handler = TimedRotatingFileHandler(
        filename=str(log_file), when="midnight", interval=1, backupCount=7, encoding="utf-8"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[handler, logging.StreamHandler()],
    )


def get_logger(name: str = None) -> logging.Logger:
    """
    Retrieve a logger instance, either by explicit name or from the caller's module context.

    Args:
        name (str, optional): Name of the logger. If None, determines name from calling module.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if name:
        return logging.getLogger(name)

    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    return logging.getLogger(module.__name__ if module else "__main__")
