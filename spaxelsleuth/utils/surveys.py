from enum import Enum, auto
import numpy as np
from dataclasses import dataclass
from typing import List

class BinTypes(Enum):
    DEFAULT = auto()
    ADAPTIVE = auto()
    SECTORS = auto()

class Ncomponents(Enum):
    ONE = auto()
    RECOM = auto()

# Survey classes 
@dataclass
class GenericSurvey:
    as_per_px: float 
    eline_list: List[float]
    sigma_inst_kms: float

@dataclass
class Sami:
    bin_type: BinTypes
    ncomponents: Ncomponents
    as_per_px = 0.5 
    eline_list = ["HALPHA", "HBETA", "NII6583", "OI6300", "OII3726+OII3729", "OIII5007", "SII6716", "SII6731"]
    sigma_inst_kms = 29.6


# Usage: 
# from surveys import Sami, SamiBinTypes
# survey = Sami(bin_type=SamiBinTypes.ADAPTIVE)