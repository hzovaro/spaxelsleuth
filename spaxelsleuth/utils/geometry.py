import numpy as np

import logging 
logger = logging.getLogger(__name__)

###############################################################################
def deproject_coordinates(x_c_list,
                          y_c_list,
                          x0_px,
                          y0_px,
                          PA_deg,
                          i_deg,
                          plotit=False,
                          im=None):
    """Deproject coordinates x_c_list, y_c_list given a galaxy inclination (i_deg), PA (PA_deg) and centre coordinates (x0_px, y0_px).
       If plotit is set to True, creates a plot showing the projected and de-projected coordinates overlaid onto the provided image im.
    """
    logger.debug(f"deprojecting coordinates...")
    i_rad = np.deg2rad(i_deg)
    beta_rad = np.deg2rad(PA_deg - 90)

    # De-project the centroids to the coordinate system of the galaxy plane
    x_prime_list = np.full_like(x_c_list, np.nan, dtype="float")
    y_prime_list = np.full_like(x_c_list, np.nan, dtype="float")
    y_prime_projec_list = np.full_like(x_c_list, np.nan, dtype="float") #NOTE I'm not sure why I calculated this?
    r_prime_list = np.full_like(x_c_list, np.nan, dtype="float")
    for jj, coords in enumerate(zip(x_c_list, y_c_list)):
        x_c, y_c = coords
        # De-shift, de-rotate & de-incline
        x_cc = x_c - x0_px
        y_cc = y_c - y0_px
        x_prime = x_cc * np.cos(beta_rad) + y_cc * np.sin(beta_rad)
        y_prime_projec = (- x_cc * np.sin(beta_rad) + y_cc * np.cos(beta_rad))
        y_prime = (- x_cc * np.sin(beta_rad) + y_cc * np.cos(beta_rad)) / np.cos(i_rad)
        r_prime = np.sqrt(x_prime**2 + y_prime**2)

        # Add to list
        x_prime_list[jj] = x_prime
        y_prime_list[jj] = y_prime
        y_prime_projec_list[jj] = y_prime_projec
        r_prime_list[jj] = r_prime

    # For plotting
    if plotit:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
        axs[0].imshow(im, origin="lower")
        axs[1].axhline(0)
        axs[1].axvline(0)
        axs[0].scatter(x_c_list, y_c_list, color="k")
        axs[0].scatter(x0_px, y0_px, color="white")
        axs[1].scatter(x_prime_list, y_prime_list, color="r")
        axs[1].scatter(x_prime_list, y_prime_projec_list, color="r", alpha=0.3)
        axs[1].axis("equal")
        fig.canvas.draw()

    return x_prime_list, y_prime_list, r_prime_list