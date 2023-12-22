import json
import multiprocessing
from pathlib import Path
import pkgutil
import os

from IPython.core.debugger import set_trace

import logging
logging.captureWarnings(True)
logger = logging.getLogger(__name__)

def configure_multiprocessing():
    """Configure multiprocessing to use 'fork' rather than 'spawn' to prevent reinitialising the 'settings' global variable when running on OSX."""
    multiprocessing.set_start_method("fork")
    return

def configure_logger(logfile_name=None, level="INFO"):
    """Configure the logger for spaxelsleuth."""
    if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        raise ValueError(f"level must be one of the following: 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'")
    if logfile_name is None:
        logging.basicConfig(
            format='%(filename)s (%(lineno)s) %(funcName)s(): %(levelname)s: %(message)s', 
            level=logging.getLevelName(level),
            force=True)
    else:
        logging.basicConfig(
            filename=logfile_name, filemode="w",
            format='%(filename)s (%(lineno)s) %(funcName)s(): %(levelname)s: %(message)s', 
            level=logging.getLevelName(level),
            force=True)

def load_default_config():
    """Load the default config file."""
    config_file_fname = Path(pkgutil.get_loader(__name__).get_filename()).parent / "config.json"
    logger.info(f"loading default config file from {config_file_fname}...")
    with open(config_file_fname, "r") as f:
        global settings
        settings = json.load(f)

def update_dictionary(d1, d2):
    """Recursive function for updating nested dictionaries with arbitrary depth."""
    for k, v in d2.items():
        if isinstance(v, dict):
            # Add entry to d1 if it does not exist
            if k not in d1:
                d1[k] = {}
            d1[k] = update_dictionary(d1=d1[k], d2=v)
        else:
            d1[k] = v
    return d1

def load_user_config(p):
    #TODO check for verbose arg throughout scripts etc.
    """Load a custom config file. Overwrites default configuration settings."""
    # Load user settings
    logger.info(f"loading user config file from from {p}...")
    with open(Path(p)) as f:
        user_settings = json.load(f)
    
    # Merge with existing settings
    global settings
    settings = update_dictionary(settings, user_settings)