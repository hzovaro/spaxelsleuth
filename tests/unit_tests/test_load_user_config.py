from copy import deepcopy
import json

from spaxelsleuth import load_user_config, configure_logger
configure_logger(level="INFO")
from spaxelsleuth.config import update_dictionary, settings

import logging
logger = logging.getLogger(__name__)


def test_load_user_config():
    """Test the behaviour of load_user_config."""
    # Test case 1: test on some dummy data 
    x = {
        "Name": "Johnny",
        "Age": 20,
        "Bees": {
            "Blue-banded": 3,
            "Hornet": 7,
        },
    }
    y = {
        "Name": "Johnny B. Goode",
        "Height": 1.73,
        "Bees": {
            "Stingless": 27,
            "Stingless_params": {
                "avg_length": 10.5,
                "avg_weight": 0.32,
            }
        },
    }
    x_updated = update_dictionary(deepcopy(x), y)
    assert x_updated["Height"] == y["Height"]
    assert x_updated["Name"] == y["Name"]
    assert x_updated["Bees"] == {
        "Blue-banded": 3,
        "Hornet": 7,
        "Stingless": 27,
        "Stingless_params": {
                "avg_length": 10.5,
                "avg_weight": 0.32,
            }
    }

    # Test case 2: test on realistic spaxelsleuth configs
    # Make a temporary file for testing 
    user_settings = {
        "plotting": {
            "log N2": {
                "vmin": -2.0,
                "some_other_variable": 999,
                "my_dict": {
                    "Hello": "world",
                }
            }
        }
    }
    settings_updated = update_dictionary(deepcopy(settings), user_settings)
    assert settings_updated["plotting"]["log N2"]["vmin"] == -2.0
    assert settings_updated["plotting"]["log N2"]["some_other_variable"] == 999
    assert settings_updated["plotting"]["log N2"]["my_dict"] == {
                    "Hello": "world",
                }

    # Test case 3: test in load_user_config
    with open(".test_config.json", "w") as f:
        json.dump(user_settings, f)
    load_user_config(".test_config.json")
    from spaxelsleuth.config import settings as settings_updated_2
    assert settings_updated_2["plotting"]["log N2"]["vmin"] == -2.0
    assert settings_updated_2["plotting"]["log N2"]["some_other_variable"] == 999
    assert settings_updated_2["plotting"]["log N2"]["my_dict"] == {
                    "Hello": "world",
                }
    
    logger.info("All test cases passed!")

    return

if __name__ == "__main__":
    test_load_user_config()
