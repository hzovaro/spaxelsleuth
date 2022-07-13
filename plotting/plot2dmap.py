"""
File:       plot2dmap.py
Author:     Henry Zovaro
Email:      henry.zovaro@anu.edu.au

DESCRIPTION
------------------------------------------------------------------------------
Contains the following functions:

    plot2dmap():
        Plot a reconstructed 2D map of the quantity specified by col_z in a 
        single galaxy, e.g. gas velocity dispersion.

------------------------------------------------------------------------------
Copyright (C) 2022 Henry Zovaro 
"""
################################################################################
# Imports
import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

from spaxelsleuth.plotting.plottools import vmin_fn, vmax_fn, label_fn, cmap_fn 
from spaxelsleuth.plotting.plottools import plot_scale_bar, plot_compass
from spaxelsleuth.plotting.plottools import bpt_ticks, bpt_labels, whav_ticks, whav_labels, morph_ticks, morph_labels, law2021_ticks, law2021_labels, ncomponents_ticks, ncomponents_labels

import matplotlib.pyplot as plt
plt.ion()

from IPython.core.debugger import Tracer

###############################################################################
def plot2dmap(df_gal, col_z, bin_type, survey,
              PA_deg=0,
              show_title=True, axis_labels=True,
              vmin=None, vmax=None, cmap=None,
              contours=True, col_z_contours="continuum", levels=None, linewidths=0.5, colors="white",
              ax=None, plot_colorbar=True, cax=None, cax_orientation="vertical",
              figsize=(5, 5)):
    """
    Show a reconstructed 2D map of the quantity specified by col_z in a single 
    galaxy.

    INPUTS
    ---------------------------------------------------------------------------
    df_gal:         pandas DataFrame
        DataFrame containing spaxel-by-spaxel data for a single galaxy.

    col_z:              str
        Quantity used to colour the image. Must be a column in df.
        NOTE: if you want to plot discrete quantities, such as BPT category,
        then you must specify the numeric option for these, i.e. set 
        col_z = "BPT (numeric)" rather than "BPT".
    
    bin_type:           str 
        The binning scheme (if any) that was used to derive the quantities in
        df_gal. Must be one of 'adaptive' or 'default' or 'sectors'.
    
    survey:             str
        The survey from which the data was derived. This must be specified 
        due to the different formats of the FITS files that are used in this 
        routine to plot continuum contours and to derive the WCS. Must be 
        either 'sami' or 's7'.

    PA_deg:             float 
        Position angle of the observations on which the data in df_gal is 
        based.

    show_title:         bool
        If True, adds a title to the axes showing the galaxy's ID.

    axis_labels:        bool
        Whether to apply axis labels as returned by label_fn(col_<x/y>) in 
        plottools.py.        

    vmin:               float
        Minimum value of col_z to use in the colour scale for the image. 
        Defaults to vmin_fn(col_z) in plottools.py.
    
    vmax:               float
        Maximum value of col_z to use in the colour scale for the image. 
        Defaults to vmax_fn(col_z) in plottools.py.
    
    cmap:               str
        Matplotlib colourmap to use. Defaults cmap_fn(col_z) in plottools.py.

    contours:           bool
        If True, overlays contours corresponding to the quantity col_z_contours 
        on top of the image.

    col_z_contours:     str
        Quantity . Must either be "continuum", in which case the mean B-band
        image extracted from the original data cube is used, which is obtained
        by computing the mean of the datacube in the rest-frame wavelength range 
        4000Å - 5000Å.

    levels:             Numpy array
        Contour levels.

    linewidths:         float 
        Contour linewidths.

    colors:             str
        Contour colours.

    ax:                 matplotlib.axis
        Axis in which to plot colourbar if plot_colorbar is True. Note that 
        because axis projections cannot be changed after an axis is created, 
        the original axis is removed and replaced with one of the same size with
        the correct WCS projection. As a result, the order of the axis in 
        fig.get_axes() may change! 

    plot_colorbar:      bool
        Whether to plot a colourbar.
    
    cax:                matplotlib.axis
        Axis in which to plot colourbar if plot_colorbar is True. If no axis 
        is specified, a new colourbar axis is created to the side of the 
        main figure axis.
    
    cax_orientation:    str
        Colourbar orientation. May be "vertical" (default) or "horizontal".

    figsize:            tuple (width, height)
        Figure size in inches.

    OUTPUTS
    ---------------------------------------------------------------------------
    Returns a tuple containing the matplotlib figure object that is the parent of 
    the main axis and the main axis.

    """
    ###########################################################################
    # Input verification
    ###########################################################################
    assert col_z in df_gal.columns,\
        f"{col_z} is not a valid column!"
    assert df_gal[col_z].dtype != "O",\
        f"{col_z} has an object data type - if you want to use discrete quantities, you must use the 'numeric' format of this column instead!"
    assert cax_orientation == "horizontal" or cax_orientation == "vertical",\
        "cax_orientation must be either 'horizontal' or 'vertical'!"
    assert len(df_gal.catid.unique()) == 1,\
        "df_gal contains must only contain one galaxy!"
    
    survey = survey.lower()
    assert survey in ["sami", "s7"],\
        "survey must be either SAMI or S7!"
    if survey == "sami":
        assert "SAMI_DIR" in os.environ, "Environment variable SAMI_DIR is not defined!"
        sami_data_path = os.environ["SAMI_DIR"]
        assert "SAMI_DATACUBE_DIR" in os.environ, "Environment variable SAMI_DATACUBE_DIR is not defined!"
        sami_datacube_path = os.environ["SAMI_DATACUBE_DIR"]
        assert bin_type in ["adaptive", "default", "sectors"],\
        "bin_type must be either 'adaptive' or 'default' or 'sectors'!"
        as_per_px = 0.5
    elif survey == "s7":
        assert "S7_DIR" in os.environ, "Environment variable S7_DIR is not defined!"
        s7_data_path = os.environ["S7_DIR"]
        assert bin_type == "default",\
        "if survey is S7 then bin_type must be 'default'!"
        as_per_px = 1.0

    ###########################################################################
    # Load the data cube to get the WCS and continuum image, if necessary
    ###########################################################################
    # Load the non-binned data cube to get a continuum image
    gal = df_gal.catid.unique()[0]
    if survey == "sami":
        gal = int(gal)
        hdulist = fits.open(os.path.join(sami_datacube_path, f"ifs/{gal}/{gal}_A_cube_blue.fits.gz"))
    elif survey == "s7":
        hdulist = fits.open(os.path.join(s7_data_path, f"0_Cubes/{gal}_B.fits"))

    # Get the WCS
    wcs = WCS(hdulist[0].header).dropaxis(2)
    if survey == "s7":
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # Get the continuum inage if desired
    if col_z_contours.lower() == "continuum":
        data_cube = hdulist[0].data
        header = hdulist[0].header
        crpix = header["CRPIX3"] if survey == "sami" else 0
        lambda_0_A = header["CRVAL3"] - crpix * header["CDELT3"]
        dlambda_A = header["CDELT3"]
        N_lambda = header["NAXIS3"]
        
        lambda_vals_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A 
        start_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + df_gal["z_spec"].unique()[0]) - 4000))
        stop_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + df_gal["z_spec"].unique()[0]) - 5000))
        im_B = np.nansum(data_cube[start_idx:stop_idx], axis=0)
        im_B[im_B == 0] = np.nan

    hdulist.close()

    ###########################################################################
    # Reconstruct & plot the 2D map
    ###########################################################################
    # Reconstruct 2D arrays from the rows in the data frame.
    if survey == "sami":
        col_z_map = np.full((50, 50), np.nan)
    elif survey == "s7":
        col_z_map = np.full((38, 25), np.nan)
    
    if bin_type == "adaptive" or bin_type == "sectors":
        hdulist = fits.open(os.path.join(sami_data_path, f"ifs/{gal}/{gal}_A_{bin_type}_blue.fits.gz"))
        bin_map = hdulist[2].data.astype("float")
        bin_map[bin_map==0] = np.nan
        for nn in df_gal["Bin number"]:
            bin_mask = bin_map == nn
            col_z_map[bin_mask] = df_gal.loc[df_gal["Bin number"] == nn, col_z]

    elif bin_type == "default":
        pd.options.mode.chained_assignment = None
        df_gal["x, y (pixels)"] = list(zip(df_gal["x (projected, arcsec)"] / as_per_px, df_gal["y (projected, arcsec)"] / as_per_px))
        pd.options.mode.chained_assignment = "warn"
        for rr in range(df_gal.shape[0]):
            xx, yy = [int(cc) for cc in df_gal.iloc[rr]["x, y (pixels)"]]
            col_z_map[yy, xx] = df_gal.iloc[rr][col_z]

    # If no axis is specified then create a new one with a vertical colorbar.
    if ax is None:
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, subplot_kw={"projection": wcs})
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.1, 0.1, 0.6, 0.8], projection=wcs)
    else:
        # Sneaky... replace the provided axis with one that has the correct projection
        fig = ax.get_figure()
        bbox = ax.get_position()
        ax.remove()
        ax = fig.add_axes(bbox, projection=wcs)

    # Plot.
    if vmin is None:
        vmin = vmin_fn(col_z)
    if vmax is None:
        vmax = vmax_fn(col_z)
    if cmap is None:
        cmap = cmap_fn(col_z)
        cmap.set_bad("#b3b3b3")
    elif type(cmap) == str:
        cmap = plt.cm.get_cmap(cmap)
        cmap.set_bad("#b3b3b3")
    m = ax.imshow(col_z_map, cmap=cmap, vmin=vmin, vmax=vmax)

    ###########################################################################
    # Contours
    ###########################################################################
    if contours:
        if col_z_contours.lower() == "continuum":
            if survey == "sami":
                ax.contour(im_B, linewidths=linewidths, colors=colors, levels=10 if levels is None else levels)
            elif survey == "s7":
                ax.contour(np.log10(im_B) + 15, linewidths=linewidths, colors=colors, levels=10 if levels is None else levels)
        else:
            assert col_z_contours in df_gal.columns, f"{col_z_contours} not found in df_gal!"
            # Reconstruct 2D arrays from the rows in the data frame.
            col_z_contour_map = np.full((50, 50), np.nan) if survey == "sami" else np.full((38, 25), np.nan)
            if bin_type == "adaptive" or bin_type == "sectors":
                hdulist = fits.open(os.path.join(sami_data_path, f"ifs/{gal}/{gal}_A_{bin_type}_blue.fits.gz"))
                bin_map = hdulist[2].data.astype("float")
                bin_map[bin_map==0] = np.nan
                for nn in df_gal["Bin number"]:
                    bin_mask = bin_map == nn
                    col_z_contour_map[bin_mask] = df_gal.loc[df_gal["Bin number"] == nn, col_z_contours]

            elif bin_type == "default":
                df_gal["x, y (pixels)"] = list(zip(df_gal["x (projected, arcsec)"] / as_per_px, df_gal["y (projected, arcsec)"] / as_per_px))
                for rr in range(df_gal.shape[0]):
                    xx, yy = [int(cc) for cc in df_gal.iloc[rr]["x, y (pixels)"]]
                    col_z_contour_map[yy, xx] = df_gal.iloc[rr][col_z_contours]

            # Draw contours
            ax.contour(col_z_contour_map, linewidths=linewidths, colors=colors, levels=10 if levels is None else levels)

    ###########################################################################
    # Colourbar
    ###########################################################################
    # If the user wants to plot a colorbar but the colorbar axis is not specified,
    # then create a new one.
    if plot_colorbar and cax is None:
        bbox = ax.get_position()
        if cax_orientation == "vertical":
            cax = fig.add_axes([bbox.x0 + bbox.width, bbox.y0, bbox.width * 0.1, bbox.height])
        elif cax_orientation == "horizontal":
            cax = fig.add_axes([bbox.x0, bbox.y0 + bbox.height, bbox.width, bbox.height * 0.1])

    # Add labels & ticks to colourbar, if necessary
    if plot_colorbar:
        plt.colorbar(mappable=m, cax=cax, orientation=cax_orientation)
        if cax_orientation == "vertical":
            cax.set_ylabel(label_fn(col_z))
        elif cax_orientation == "horizontal":
            cax.xaxis.set_ticks_position("top")
            cax.xaxis.set_label_position('top')
            cax.set_xlabel(label_fn(col_z))
        if col_z.startswith("BPT (numeric)"):
            if cax_orientation == "vertical":
                cax.yaxis.set_ticks(bpt_ticks)
                cax.yaxis.set_ticklabels(bpt_labels)
            elif cax_orientation == "horizontal":
                cax.xaxis.set_ticks(bpt_ticks)
                cax.xaxis.set_ticklabels(bpt_labels)
        if col_z.startswith("WHAV* (numeric)"):
            if cax_orientation == "vertical":
                cax.yaxis.set_ticks(whav_ticks)
                cax.yaxis.set_ticklabels(whav_labels)
            elif cax_orientation == "horizontal":
                cax.xaxis.set_ticks(whav_ticks)
                cax.xaxis.set_ticklabels(whav_labels)
        if col_z == "Morphology (numeric)":
            if cax_orientation == "vertical":
                cax.yaxis.set_ticks(morph_ticks)
                cax.yaxis.set_ticklabels(morph_labels)
            elif cax_orientation == "horizontal":
                cax.xaxis.set_ticks(morph_ticks)
                cax.xaxis.set_ticklabels(morph_labels)
        if col_z.startswith("Law+2021 (numeric)"):
            if cax_orientation == "vertical":
                cax.yaxis.set_ticks(law2021_ticks)
                cax.yaxis.set_ticklabels(law2021_labels)
            elif cax_orientation == "horizontal":
                cax.xaxis.set_ticks(law2021_ticks)
                cax.xaxis.set_ticklabels(law2021_labels)
        if col_z == "Number of components":
            if cax_orientation == "vertical":
                cax.yaxis.set_ticks(ncomponents_ticks)
                cax.yaxis.set_ticklabels(ncomponents_labels)
            elif cax_orientation == "horizontal":
                cax.xaxis.set_ticks(ncomponents_ticks)
                cax.xaxis.set_ticklabels(ncomponents_labels)

    ###########################################################################
    # Decorations
    ###########################################################################
    # Include scale bar
    plot_scale_bar(as_per_px=as_per_px, kpc_per_as=df_gal["kpc per arcsec"].unique()[0], fontsize=10, ax=ax, l=10, units="arcsec", color="black", loffset=0.30)
    plot_compass(ax=ax, color="black", PA_deg=PA_deg)

    # Title
    if show_title:
        ax.set_title(f"GAMA{gal}") if survey == "sami" else ax.set_title(gal)

    # Axis labels
    if axis_labels:
        ax.set_ylabel("Dec (J2000)")
        ax.set_xlabel("RA (J2000)")

    return fig, ax


