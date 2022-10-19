"""

1. emission line fluxes 
    - Halpha: for each component
    - other lines: total fluxes only
2. emission line amplitude requirement 
    - we can only do this for Halpha 
    - we should assume that if the Halpha doesn't meet this requirement then
      no other lines do, as it is typically one of the strongest lines
    --> CHANGED: now NaNing out ALL columns containing "component {nn + 1}" (except for flags)

How to apply these cuts:
    - total emission line fluxes: only cut if total S/N in that emission line is low
        Don't care about kinematics
    - Halpha EW: 
    - 

Other cuts:
- stellar kinematics:
    --> NaN out stellar kinematics 
    --> can still keep fluxes

- vgrad:
    --> NaN out gas kinematics (per component)
    --> can still keep fluxes
    --> CHANGED: now NaNs out all columns containing v_gas or sigma_gas

- sigma_gas S/N:
    --> NaN out gas velocity dispersion (per component)
    --> can still keep fluxes
    --> CHANGED: now NaNs out all columns containing sigma_gas

"""