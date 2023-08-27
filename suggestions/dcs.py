from dataclasses import dataclass

@dataclass
class Spaxel:
    ok_stellar_kinematics: bool
    missing_components: bool
    elines: list[set[EmissionLine]]
    components: list[set[Component]]

@dataclass
class Galaxy:
    name: str
    spaxels: Spaxels

@dataclass
class Spaxels:
    ok_stellar_kinematics: np.ndarray[bool]

@dataclass
class EmissionLineComponent:
    index: int  # from zero
    flux: float
    flux_err: float
    v_gas: float
    v_gas_err: float
    beam_smear: bool
    sigma_gas: float
    sigma_gas_err: float
    low_sigma_gas_sn: bool
    lambda_obs: float
    low_flux_fraction: bool
    low_amp: bool
    missing_flux: bool
    low_flux_sn: bool

@dataclass
class EmissionLine:
    flux_tot: float
    flux_tot_err: float
    components: set[EmissionLineComponent]
    low_flux_fraction: bool
    low_amp: bool
    missing_flux: bool
    low_flux_sn: bool

@dataclass
class Component:
    v_gas: float
    v_gas_err: float
    beam_smear: bool
    sigma_gas: float
    sigma_gas_err: float
