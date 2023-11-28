__version__="0.9.0"
__authors__=["Henry R. M. Zovaro"]
__date__="2023-09-11"

# Load the default configuration file
from .config import *
configure_logger()
print_directory()
load_default_config()
configure_multiprocessing()