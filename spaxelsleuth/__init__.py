__version__="1.1.0"
__authors__=["Henry R. M. Zovaro"]
__date__="2024-02-16"

# Load the default configuration file
from .config import *
configure_logger()
load_default_config()
configure_multiprocessing()