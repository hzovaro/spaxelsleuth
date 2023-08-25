import logging
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(filename)s (line %(lineno)s) %(funcName)s(): %(message)s', level=logging.DEBUG, force=True)

def myfunction(x):
    local_logger = logging.getLogger(__name__)

    local_logger.info("Incrementing x by 1")
    x += 1
    local_logger.info("Computing sine of x")
    s = np.sin(x)
    local_logger.info("Done")
    local_logger.debug(f"The value of x is {x:.2f}")
    return x, s

logger.info("Calling myfunction()")
a, b = myfunction(np.pi / 3)