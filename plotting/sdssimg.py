import os
import numpy as np
from urllib.request import urlretrieve
from astropy.wcs import WCS

from cosmocalc import get_dist
from plotting_fns import plot_scale_bar

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle

from IPython.core.debugger import Tracer

SDSS_IM_PATH = "/priv/meggs3/u5708159/SAMI/sami_dr3/sdss/"

###############################################################################
def get_sdss_image(gal, ra_deg, dec_deg,
                   as_per_px=0.1, width_px=500, height_px=500):
    """
    Download an SDSS cutout image.
    """
    
    # Determine the URL
    url = f"http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra={ra_deg}&dec={dec_deg}&scale={as_per_px:.1f}&width={width_px}&height={height_px}&opt=G"
    
    # Download the image
    imname = os.path.join(SDSS_IM_PATH, f"{gal}_{width_px}x{height_px}.jpg")
    
    try:
        urlretrieve(url, imname)
    except:
        print(f"{gal} not in SDSS footprint!")
        return False

    return True


###############################################################################
def plot_sdss_image(df_gal, show_title=True, axis_labels=True,
                    as_per_px=0.1, width_px=500, height_px=500,
                    ax=None, figsize=(9, 7)):    

    # Input checking
    assert len(df_gal["catid"].unique()) == 1, "df_gal must only contain one galaxy!!"

    # Get the central coordinates from the DF
    ra_deg = df_gal["ra_obj"].unique()[0]
    dec_deg = df_gal["dec_obj"].unique()[0]
    gal = df_gal["catid"].unique()[0]

    # Load image
    if not os.path.exists(os.path.join(SDSS_IM_PATH, f"{gal}_{width_px}x{height_px}.jpg")):
        # Download the image
        print(f"WARNING: file {os.path.join(SDSS_IM_PATH, f'{gal}_{width_px}x{height_px}.jpg')} not found. Retrieving image from SDSS...")
        if not get_sdss_image(gal=gal, ra_deg=ra_deg, dec_deg=dec_deg,
                       as_per_px=as_per_px, width_px=width_px, height_px=height_px):
            return None
            
    im = mpimg.imread(os.path.join(SDSS_IM_PATH, f"{gal}_{width_px}x{height_px}.jpg"))

    # Make a WCS for the image
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [im.shape[0] // 2, im.shape[1] // 2]
    wcs.wcs.cdelt = np.array([0.1 / 3600, 0.1 / 3600])
    wcs.wcs.crval = [ra_deg, dec_deg]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # If no axis is specified then create a new one with a vertical colorbar.
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, subplot_kw={"projection": wcs})
    else:
        # Sneaky... replace the provided axis with one that has the correct projection
        fig = ax.get_figure()
        bbox = ax.get_position()
        ax.remove()
        ax = fig.add_axes(bbox, projection=wcs)

    # Display the image
    ax.imshow(np.flipud(im))
    """
    WEIRD PROBLEM: only occurs for SOME galaxies. Exception occurs at 
        ax.imshow(np.flipud(im))
    also occurs when plotting a different image, e.g.
        ax.imshow(np.random.normal(loc=0, scale=10, size=(500, 500)))
    appears to be a latex error. Doesn't occur if the axis is not a WCS
    """
    # Overlay a circle w/ diameter 15 arcsec
    c = Circle((ra_deg, dec_deg), radius=7.5 / 3600, transform=ax.get_transform("world"), facecolor='none', edgecolor="w", lw=1, ls="-", zorder=999999)
    ax.add_patch(c)

    # Include scale bar
    D_A_Mpc, D_L_Mpc = get_dist(z=df_gal["z_spec"].unique()[0], H0=70.0, WM=0.3)
    kpc_per_as = D_A_Mpc * 1e3 * np.pi / 180.0 / 3600.0
    plot_scale_bar(as_per_px=0.1, loffset=0.25, kpc_per_as=kpc_per_as, ax=ax, l=10, units="arcsec", fontsize=10)

    # Axis labels
    if axis_labels:
        ax.set_ylabel("Dec (J2000)")
        ax.set_xlabel("RA (J2000)")
    
    return ax