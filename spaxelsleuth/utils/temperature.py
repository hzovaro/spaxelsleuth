import numpy as np

from spaxelsleuth.utils.misc import remove_col_suffix, add_col_suffix

import logging
logger = logging.getLogger(__name__)


def get_T_e_Proxauf2014(R):
    """"
    Electron temperature computation from eqn. 1 of Proxauf et al. (2014).
    Valid for electron densitites up to a few 10^4 cm^-3. 

    INPUT
    --------------------------------------------------------------------------
    R:          float
        Flux ratio F([OIII]4959,5007) / F([OIII]4363).

    OUTPUT
    --------------------------------------------------------------------------
    tuple of 
        (T_e, lolim_mask, uplim_mask)
    where T_e is the electron temperature (in K) computed using eqn. 1 of 
    Proxauf (2014), and lolim_mask and uplim_mask areboolean arrays the 
    same dimensions as n_e corresponding to indices where the line ratio 
    corresponds to electron temperature lower and upper limits of 5000 K and 
    24000 K respectively.

    REFERENCES
    --------------------------------------------------------------------------  
    https://ui.adsabs.harvard.edu/abs/2014A%26A...561A..10P/abstract
    """
    # Repalce infs with NaNs
    R = R.copy()
    R[np.isinf(R)] = np.nan

    # Calculate the electron temperature
    r = np.log10(R)
    T_e = 5294 * (r - 0.848)**(-1) + 19047 - 7769 * r + 944 * r**2

    # Apply upper/lower limits
    r_min = 1.25
    r_max = 3.75
    T_e_min = 5e3
    T_e_max = 24e3
    lolim_mask = (r > r_max)
    uplim_mask = (r < r_min)
    T_e[lolim_mask] = T_e_min
    T_e[uplim_mask] = T_e_max            

    return T_e, lolim_mask, uplim_mask


def get_T_e_PM2014(ratio, R):
    """"
    Electron temperature computation from eqn. 1 of Perez Montero (2014).
    Assumes a density of 100 cm^-3. Note that the returned temperatures 
    reflect the temperature associated with the ions corresponding to the 
    chosen line ratios. 

    INPUT
    --------------------------------------------------------------------------
    ratio:      str
        Which emission line to use in the diagnostic. Options are "[NII]" and
        "[OIII]". 
        
        NOTE: the [SIII] diagnostic included in this paper has not 
        been implemented due to a suspected typo in their eqn. 11; specifying
        the [SIII] ratio will raise a NotImplementedError. 
    
    R:          float
        Flux ratio. Must correspond to 'ratio':
            [NII]: F([NII]6548,6584) / F([NII]5755) 
            [OII]: F([OIII]4959,5007) / F([OIII]4363) 
            [SIII]: F([SIII]9069,9531) / F([SIII]6312) 

    OUTPUT
    --------------------------------------------------------------------------
    tuple of 
        (T_e, lolim_mask, uplim_mask)
    where T_e is the electron temperature (in K) computed using eqns. 5 and 7 
    of Perez Montero (2014), and lolim_mask and uplim_mask are boolean arrays 
    the same dimensions as n_e corresponding to indices where the line ratio 
    corresponds to valid electron temperatures which are in the range 
        * (0.7, 2.5) * 10^4 K for the [OIII] diagnostic,
        * (0.6, 2.2) * 10^4 K for the [NII] diagnostic, and 
        * (0.6, 2.5) * 10^4 K for the [SIII] diagnostic.

    REFERENCES
    --------------------------------------------------------------------------  
    https://ui.adsabs.harvard.edu/abs/2014MNRAS.441.2663P/abstract
    """
    # Repalce infs with NaNs
    R = R.copy()
    R[np.isinf(R)] = np.nan

    # Calculate the electron temperature
    if ratio == "[OIII]":
        T_e = (0.7840 - 0.0001357 * R + 48.44 / R) * 1e4
        T_e_min = 0.7e4
        T_e_max = 2.5e4
    elif ratio == "[NII]":
        T_e = (0.6153 - 0.0001529 * R + 35.3641 / R) * 1e4
        T_e_min = 0.6e4
        T_e_max = 2.2e4
    elif ratio == "[SIII]":
        raise NotImplementedError("For now, the [SIII] diagnostic has not been implemented due to a suspected typo in their eqn. 11")
        """
        If you plot the below equation, which is as printed in the paper (eqn. 11),
        as a function of R, you will see that it has a minimum at approx. R = 300
        and is therefore two-valued within the specified range of valid temperatures.
        I suspect that the plus sign before 0.0003187 is actually meant to be a minus
        sign, which would make it consistent with the other two diagnostics. Indeed,
        if it is replaced with a minus sign the minimum disappears and the values look
        more reasonable. 
        """
        T_e = (0.5147 + 0.0003187 * R + 23.64041 / R) * 1e4
        T_e_min = 0.6e4
        T_e_max = 2.5e4

    else:
        raise ValueError(f"{ratio} is not a valid line ratio! Valid ratios are [OIII] and [NII]!")

    lolim_mask = (T_e < T_e_min)
    uplim_mask = (T_e > T_e_max)
    T_e[lolim_mask] = T_e_min
    T_e[uplim_mask] = T_e_max            

    return T_e, lolim_mask, uplim_mask


def compute_electron_temperature(df, diagnostic, ratio="[OIII]", s=None):
    """
    Calculate electron temperature using emission line ratios of [OIII].

    INPUTS
    --------------------------------------------------------------------------
    df:         pandas DataFrame
        DataFrame in which to compute the electron temperature.

    diagnostic: str
        Which diagnostic to use. Options are "Proxauf2014" and "PM2014".

    ratio:      str
        Which emission line to use in the diagnostic. Valid options are
        * "[OIII]" for the "Proxauf2014" diagnostic or 
        * "[OIII]", "[NII]" or "[SIII]" for the "PM2014" diagnostic.        

    s:          str 
        Column suffix to trim before carrying out computation - e.g. if 
        you want to compute metallicities for "total" fluxes, and the 
        columns of the DataFrame look like 

            "HALPHA (total)", "HALPHA error (total)", etc.,

        then setting s=" (total)" will mean that this function "sees"

            "HALPHA", "HALPHA error".

        Useful for running this function on different emission line 
        components. The suffix is added back to the columns (and appended
        to any new columns that are added) before being returned. For 
        example, using the above example, the new added columns will be 

            "T_e (<diagnostic>) (<ratio>) (total)" and
            "T_e saturation flag (<diagnostic>) (<ratio>) (total)"

    OUTPUTS
    -----------------------------------------------------------------------
    The original DataFrame with new columns added:

        "T_e (<diagnostic>) (<ratio>) <suffix>": electron temperature in K
        
        "T_e saturation flag (<diagnostic>) (<ratio>) <suffix>": True if 
        the line ratios correspond to "saturation" in the diagnostic used, 
        False otherwise. T_e measurements corresponding to "saturated" line 
        ratios will be set to the upper/lower limits for the diagnostic.

    REFERENCES
    --------------------------------------------------------------------------  
    https://ui.adsabs.harvard.edu/abs/2014A%26A...561A..10P/abstract
    https://ui.adsabs.harvard.edu/abs/2016ApJ...816...23S/abstract 
    """
    
    # Input checking 
    if diagnostic not in ["Proxauf2014", "PM2014"]:
        raise ValueError(f"{diagnostic} is an invalid diagnostic!")
    if diagnostic == "Proxauf2014":
        if ratio not in ["[OIII]"]:
            raise ValueError(f"Line ratio {ratio} is invalid for the {diagnostic} T_e diagnostic!")
    elif diagnostic == "PM2014":
        if ratio not in ["[OIII]", "[NII]", "[SIII]"]:
            raise ValueError(f"Line ratio {ratio} is invalid for the {diagnostic} T_e diagnostic!")    
    
    # Trim suffix
    df, suffix_cols, suffix_removed_cols, old_cols = remove_col_suffix(df, s)
    old_cols = df.columns

    # Calculate the electron temperature 
    if ratio == "[OIII]":
        if f"OIII4363" not in df and "OIII4959+OIII5007" not in df:
            logger.warning(f"not computing T_e because I could not find the columns OIII4363 and/or OIII4959+OIII5007")
        else:
            # Calculate 
            R = df["OIII4959+OIII5007"] / df[f"OIII4363"]
            if diagnostic == "Proxauf2014":
                T_e, cond_lower_lim, cond_upper_lim = get_T_e_Proxauf2014(R)
            elif diagnostic == "PM2014":
                T_e, cond_lower_lim, cond_upper_lim = get_T_e_PM2014(ratio="[OIII]", R=R)

    elif ratio == "[NII]":
        if "NII6548+NII6583" not in df and "NII5755" not in df:
            logger.warning(f"not computing T_e because I could not find the columns NII5755 and/or NII6548+NII6583")
        else:
            # Calculate 
            R = df["NII6548+NII6583"] / df[f"NII5755"]
            if diagnostic == "PM2014":
                T_e, cond_lower_lim, cond_upper_lim = get_T_e_PM2014(ratio="[NII]", R=R)

    elif ratio == "[SIII]":
        if "SIII9069+SIII9531" not in df and "SIII6312" not in df:
            logger.warning(f"not computing T_e because I could not find the columns SIII6312 and/or SIII9069+SIII9531")
        else:
            # Calculate 
            R = df["SIII9069+SIII9531"] / df[f"SIII6312"]
            if diagnostic == "PM2014":
                T_e, cond_lower_lim, cond_upper_lim = get_T_e_PM2014(ratio="[SIII]", R=R)

    # Store in DataFrame
    df[f"T_e ({diagnostic} ({ratio}))"] = T_e
    df[f"T_e saturation flag ({diagnostic} ({ratio}))"] = False
    df.loc[cond_lower_lim | cond_upper_lim, f"T_e saturation flag ({diagnostic} ({ratio}))"] = True

    # Rename columns
    df = add_col_suffix(df, s, suffix_cols, suffix_removed_cols, old_cols)

    return df