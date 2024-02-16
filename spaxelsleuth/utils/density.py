import numpy as np
import warnings

from spaxelsleuth.utils.misc import remove_col_suffix, add_col_suffix

import logging
logger = logging.getLogger(__name__)


def get_n_e_Proxauf2014(R):
    """
    Electron density computation from eqn. 3 of Proxauf et al. (2014), which 
    assumes an electron temperature of 10^4 K.

    INPUT
    --------------------------------------------------------------------------
    R:          float
        Flux ratio F([SII]6716) / F([SII]6731).

    OUTPUT
    --------------------------------------------------------------------------
    tuple of 
        (n_e, lolim_mask, uplim_mask)
    where n_e is the electron density (in cm^-3) computed using eqn. 3 of 
    Proxauf (2014), and lolim_mask and uplim_mask areboolean arrays the 
    same dimensions as n_e corresponding to indices where the line ratio 
    corresponds to electron density lower and upper limits of 40 cm^-3 and 
    10^4 cm^-3 respectively.

    REFERENCES
    --------------------------------------------------------------------------  
    https://ui.adsabs.harvard.edu/abs/2014A%26A...561A..10P/abstract

    """
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=RuntimeWarning)
        log_n_e = 0.0543 * np.tan(-3.0553 * R + 2.8506)\
                + 6.98 - 10.6905 * R\
                + 9.9186 * R**2 - 3.5442 * R**3
        n_e = 10**log_n_e

    # High & low density limits
    lolim_mask = n_e < 40
    uplim_mask = n_e > 1e4
    n_e[lolim_mask] = 40
    n_e[uplim_mask] = 1e4

    return n_e, lolim_mask, uplim_mask


def get_n_e_Sanders2016(ratio, R):
    """
    Electron density computation from eqn. 7 of Sanders et al. (2016), which 
    assumes an electron temperature of 10^4 K.

    INPUT
    --------------------------------------------------------------------------
    ratio:      str
        Which emission line to use in the diagnostic. Options are "[SII]" and 
        "[OII]".
    
    R:          float
        Flux ratio. Must correspond to 'ratio':
            [SII]: F([SII]6716) / F([SII]6731) 
            [OII]: F([OII]3729) / F([OII]3726) 

    OUTPUT
    --------------------------------------------------------------------------
    tuple of 
        (n_e, lolim_mask, uplim_mask)
    where n_e is the electron density (in cm^-3) computed using eqn. 7 of 
    Sanders et al. (2016), and lolim_mask and uplim_mask areboolean arrays the 
    same dimensions as n_e corresponding to indices where the line ratio 
    corresponds to electron density lower and upper limits of 1 cm^-3 and 
    10^5 cm^-3 respectively.

    REFERENCES
    --------------------------------------------------------------------------  
    https://ui.adsabs.harvard.edu/abs/2016ApJ...816...23S/abstract 

    """
    # Repalce infs with NaNs
    R = R.copy()
    R[np.isinf(R)] = np.nan

    # Coefficients 
    if ratio == "[SII]":
        a = 0.4315
        b = 2107
        c = 627.1
        R_min = 0.4375
        R_max = 1.4484
    elif ratio == "[OII]":
        a = 0.3771
        b = 2468
        c = 638.4
        R_min = 0.3839
        R_max = 1.4558
    else:
        raise ValueError(f"{ratio} is not a valid line ratio! Valid ratios are [OII] and [SII]!")

    # Calculate the electron density 
    n_e = (c * R - a * b) / (a - R)
    
    # Apply upper/lower limits
    n_e_min = 1
    n_e_max = 1e5
    lolim_mask = R > R_max
    uplim_mask = R < R_min
    n_e[lolim_mask] = n_e_min
    n_e[uplim_mask] = n_e_max
    
    return n_e, lolim_mask, uplim_mask


def compute_electron_density(df, diagnostic, ratio, s=None):
    """
    Calculate electron densities using emission line ratios of [SII] or [OII].

    INPUTS
    --------------------------------------------------------------------------
    df:         pandas DataFrame
        DataFrame in which to compute 

    diagnostic: str
        Which diagnostic to use. Options are "Proxauf2014" and "Sanders2016".

    ratio:      str
        Which emission line to use in the diagnostic. Options are "[SII]" 
        (valid for both "Proxauf2014" and "Sanders2016") and "[OII]" 
        ("Sanders2016" only).

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

            "log N2 (total)", "log N2 error (lower) (total)", 
            "log N2 error (upper) (total)"

    OUTPUTS
    -----------------------------------------------------------------------
    The original DataFrame with new columns added.

    REFERENCES
    --------------------------------------------------------------------------  
    https://ui.adsabs.harvard.edu/abs/2014A%26A...561A..10P/abstract
    https://ui.adsabs.harvard.edu/abs/2016ApJ...816...23S/abstract 
    """
    # Trim suffix
    df, suffix_cols, suffix_removed_cols, old_cols = remove_col_suffix(df, s)
    old_cols = df.columns

    if (ratio == "[SII]" and "[SII] ratio" in df) or (ratio == "[OII]" and "[OII] ratio" in df):

        # Compute diagnostics 
        if diagnostic == "Proxauf2014":
            if ratio != "[SII]":
                raise ValueError(f"Invalid ratio {ratio} for diagnostic {diagnostic}!")
            R = df["[SII] ratio"].values
            n_e, cond_lower_lim, cond_upper_lim = get_n_e_Proxauf2014(R)
        elif diagnostic == "Sanders2016":
            if ratio not in["[SII]", "[OII]"]:
                raise ValueError(f"Invalid ratio {ratio} for diagnostic {diagnostic}!")
            R = df[f"{ratio} ratio"].values
            n_e, cond_lower_lim, cond_upper_lim = get_n_e_Sanders2016(ratio, R)
        else:
            raise ValueError(f"{diagnostic} is an invalid diagnostic!")

        df[f"n_e ({diagnostic} ({ratio}))"] = n_e
        df[f"n_e saturation flag ({diagnostic} ({ratio}))"] = False
        df.loc[cond_lower_lim | cond_upper_lim, f"n_e saturation flag ({diagnostic} ({ratio}))"] = True
    
    else:
        if ratio == "[SII]":
            logger.warning(f"not computing n_e because I could not find the column '[SII] ratio'")
        elif ratio == "[OII]":
            logger.warning(f"not computing n_e because I could not find the column '[OII] ratio'")

    # Rename columns
    df = add_col_suffix(df, s, suffix_cols, suffix_removed_cols, old_cols)

    return df
