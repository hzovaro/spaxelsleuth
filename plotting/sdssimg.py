"""
File:       sdssimg.py
Author:     Henry Zovaro
Email:      henry.zovaro@anu.edu.au

DESCRIPTION
------------------------------------------------------------------------------
Contains the following functions:

    get_sdss_image():
        Download an SDSS cutout image. Used in plot_sdss_image().

    plot_sdss_image():
        Download and plot the SDSS image of a galaxy with RA and Dec in the 
        supplied pandas DataFrame. The images are stored in environment 
        variable SDSS_IM_PATH. Note that if the galaxy is outside the SDSS 
        footprint, no image is plotted.

------------------------------------------------------------------------------
Copyright (C) 2022 Henry Zovaro 
"""
################################################################################
# Imports
import os
import numpy as np
from urllib.request import urlretrieve
from astropy.wcs import WCS

from spaxelsleuth.plotting.plottools import plot_scale_bar

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle

from IPython.core.debugger import Tracer

###############################################################################
# Paths
assert "SDSS_IM_PATH" in os.environ, "Environment variable SDSS_IM_PATH is not defined!"
sdss_im_path = os.environ["SDSS_IM_PATH"]

###############################################################################
def get_sdss_image(gal, ra_deg, dec_deg,
                   as_per_px=0.1, width_px=500, height_px=500):
    """
    Download an SDSS cutout image.
    
    INPUTS
    --------------------------------------------------------------------------
    gal:            str
        Name of galaxy. Note that this is not used in actually retrieving 
        the image, and is only used in the filename of the image - hence the 
        name can be arbitrary.

    ra_deg:         float 
        Right ascension of the galaxy in degrees.

    dec_deg:        float 
        Declination of the galaxy in degrees.

    as_per_px:      float 
        Plate scale of the SDSS image in arcseconds per pixel.

    width_px:       int
        Width of image to download.

    height_px:      int
        height of image to download.

    OUTPUTS
    --------------------------------------------------------------------------
    Returns True if the image was successfully received; False otherwise.
    """
    # Determine the URL
    url = f"http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Explore.Image&ra={ra_deg}&dec={dec_deg}&scale={as_per_px:.1f}&width={width_px}&height={height_px}&opt=G"
    
    # Download the image
    imname = os.path.join(sdss_im_path, f"{gal}_{width_px}x{height_px}.jpg")
    
    try:
        urlretrieve(url, imname)
    except:
        print(f"{gal} not in SDSS footprint!")
        return False

    return True


###############################################################################
def plot_sdss_image(df_gal=None, 
                    gal=None, ra_deg=None, dec_deg=None, kpc_per_as=None,
                    axis_labels=True,
                    as_per_px=0.1, width_px=500, height_px=500,
                    reload_image=False,
                    ax=None, figsize=(5, 5)):    

    """
    Download and plot the SDSS image of a galaxy with RA and Dec in the supplied 
    pandas DataFrame. The images are stored in environment variable 
    SDSS_IM_PATH. Note that if the galaxy is outside the SDSS footprint, 
    no image is plotted.
    
    INPUTS
    --------------------------------------------------------------------------
    df_gal:         pandas DataFrame
        DataFrame containing spaxel-by-spaxel data for a single galaxy.
        Must have columns:
            ID - the catalogue ID of the galaxy
            RA (J2000) - the RA of the galaxy in degrees
            Dec (J2000) - the declination of the galaxy in degrees

    axis_labels:    bool
        If True, plot RA and Dec axis labels.

    as_per_px:      float 
        Plate scale of the SDSS image in arcseconds per pixel.

    width_px:       int
        Width of image to download.

    height_px:      int
        height of image to download.

    reload_image:   bool
        If True, force re-download of the image.

    ax:             matplotlib.axis
        axis on which to plot the image. Note that because axis projections 
        cannot be changed after an axis is created, the original axis is 
        removed and replaced with one of the same size with the correct WCS 
        projection. As a result, the order of the axis in fig.get_axes() 
        may change! 

    figsize:        tuple (width, height)
        Only used if axis is not specified, in which case a new figure is 
        created with figure size figsize.

    OUTPUTS
    --------------------------------------------------------------------------
    The axis containing the plotted image.

    """
    # Input checking
    if df_gal is not None:
        assert len(df_gal["ID"].unique()) == 1, "df_gal must only contain one galaxy!!"
        # Get the central coordinates from the DF
        if "RA (IFU) (J2000)" in df_gal and "Dec (IFU) (J2000)" in df_gal:
            ra_deg = df_gal["RA (IFU) (J2000)"].unique()[0]
            dec_deg = df_gal["Dec (IFU) (J2000)"].unique()[0]
        else:
            ra_deg = df_gal["RA (J2000)"].unique()[0]
            dec_deg = df_gal["Dec (J2000)"].unique()[0]
        gal = df_gal["ID"].unique()[0]
        kpc_per_as = df_gal["kpc per arcsec"].unique()[0]
    else:
        assert gal is not None, "gal must be specified!"
        assert ra_deg is not None,  "ra_deg must be specified!"
        assert dec_deg is not None, "dec_deg must be specified!"
        assert kpc_per_as is not None, "kpc_per_as must be specified!"

    # Load image
    if reload_image or (not os.path.exists(os.path.join(sdss_im_path, f"{gal}_{width_px}x{height_px}.jpg"))):
        # Download the image
        print(f"WARNING: file {os.path.join(sdss_im_path, f'{gal}_{width_px}x{height_px}.jpg')} not found. Retrieving image from SDSS...")
        if not get_sdss_image(gal=gal, ra_deg=ra_deg, dec_deg=dec_deg,
                       as_per_px=as_per_px, width_px=width_px, height_px=height_px):
            return None
            
    im = mpimg.imread(os.path.join(sdss_im_path, f"{gal}_{width_px}x{height_px}.jpg"))

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
    
    plot_scale_bar(as_per_px=0.1, loffset=0.25, kpc_per_as=kpc_per_as, ax=ax, l=10, units="arcsec", fontsize=10)

    # Axis labels
    if axis_labels:
        ax.set_ylabel("Dec (J2000)")
        ax.set_xlabel("RA (J2000)")
    
    return ax