# Imports
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.io import fits
from matplotlib.transforms import Affine2D
from astropy.visualization.wcsaxes import WCSAxes
from astropy.wcs import WCS

from spaxelsleuth.plotting.plottools import get_vmin, get_vmax, get_cmap, get_label, plot_scale_bar, plot_compass, trim_suffix
from spaxelsleuth.config import settings

import logging
logger = logging.getLogger(__name__)

###############################################################################
def plot2dmap(df,
              gal,
              col_z,
              bin_type=None,
              survey=None,
              PA_deg=0, #TODO get rid of this?
              as_per_px=None,
              plot_ra_dec=False,
              ra_deg=None, dec_deg=None,
              show_title=True,
              axis_labels=True,
              vmin=None,
              vmax=None,
              cmap=None,
              contours=True,
              col_z_contours="B-band continuum",
              levels=None,
              linewidths=0.5,
              colors="white",
              ax=None,
              plot_colorbar=True,
              cax=None,
              cax_orientation="vertical",
              show_compass=True,
              show_scale_bar=True,
              figsize=(5, 5)):
    """
    Show a reconstructed 2D map of the quantity specified by col_z in a single 
    galaxy.

    INPUTS
    ---------------------------------------------------------------------------
    df:                 pandas DataFrame
        DataFrame containing spaxel-by-spaxel data.

    gal:                int
        Galaxy ID. 
    
    col_z:              str
        Quantity used to colour the image. Must be a column in df.
    
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
    # Extract the subset of the DataFrame belonging to this galaxy
    df_gal = df[df["ID"] == gal]
    if df_gal.shape == 0:
        raise ValueError("df_gal is an empty DataFrame!")

    if col_z not in df_gal.columns:
        raise ValueError(f"{col_z} is not a valid column!")
    
    if df_gal[col_z].dtype == "O":
        col, suffix = trim_suffix(col_z)
        if f"{col} (numeric)" + suffix in df_gal:
            col_z = f"{col} (numeric)" + suffix
        else:
            raise ValueError(
                f"{col_z} has an object data type and no numeric counterpart exists in df!"
            )
    
    if cax_orientation not in ["horizontal", "vertical"]:
        raise ValueError(
            "cax_orientation must be either 'horizontal' or 'vertical'!")

    # Validate: survey (optional)
    # NOTE: we only need survey so that we can access the SAMI data cube path if bin_type is not default.
    if "survey" in df_gal:
        if survey is not None:
            logger.warn(f"defaulting to 'survey' found in df rather than supplied value of '{survey}'")
        if len(df_gal["survey"].unique()) > 1:
                raise ValueError(f"There appear to be multiple 'survey' values in df!")
        survey = df_gal["survey"].unique()[0]
    elif survey is not None:
        survey = survey.lower()
        if survey not in settings:
            raise ValueError(f"survey '{survey}' was not found in settings!")
    else:
        logger.warn("'survey' not specified!")
    
    # Validate: bin_type (optional)
    if "bin_type" in df_gal:
        if bin_type is not None:
            logger.warn(f"defaulting to 'bin_type' found in df rather than supplied value of '{bin_type}'")
        if len(df["bin_type"].unique()) > 1:
            raise ValueError(f"There appear to be multiple 'bin_type' values in df for galaxy {gal}!")
        bin_type = df["bin_type"].unique()[0]
        if bin_type is None:
            raise ValueError("Not sure how this happened, but the value of 'bin_type' in the DataFrame is None!")
    elif bin_type is not None:
        bin_type = bin_type.lower()
        if survey is not None: 
            if "bin_types" in settings[survey]:
                if bin_type not in settings[survey]["bin_types"]:
                    raise ValueError(f"bin_type '{bin_type}' is not valid for survey '{survey}'!")
            else:
                if bin_type != "default":
                    raise ValueError(f"bin_type must be 'default' for survey '{survey}'!")
        else:
            if bin_type != "default":
                raise ValueError(f"bin_type must be 'default' if no survey is specified!")
    else:
        logger.warn("'bin_type' not specified - assuming 'default'")
        bin_type = "default"

    ###########################################################################
    # Get geometry
    ###########################################################################
    # Think of it as: use config settings to MAKE the dataframes but store all necessary info in the dataframe itself
    if "as_per_px" in df_gal:
        as_per_px = df_gal["as_per_px"].unique()[0]
    else:
        raise ValueError("as_per_px was not found in the DataFrame!")

    # Get size of image
    if "N_x" in df_gal and "N_y" in df_gal:
        nx = int(df_gal["N_x"].unique()[0])
        ny = int(df_gal["N_y"].unique()[0])
    else:
        logger.warn("nx and ny were not found in the DataFrame so I am assuming their values from the shape of the data")
        nx = int(np.nanmax(df_gal["x (pixels)"].values) + 1)
        ny = int(np.nanmax(df_gal["y (pixels)"].values) + 1)

    # Get centre coordinates of image 
    if "x0_px" in df_gal and "y0_px" in df_gal:
        x0_px = int(df_gal["x0_px"].unique()[0])
        y0_px = int(df_gal["y0_px"].unique()[0])
    else:
        logger.warn("x0_px and y0_px were not found in the DataFrame so I am assuming their values from the shape of the data")
        x0_px = nx / 2.
        y0_px = ny / 2.

    ###########################################################################
    # Reconstruct 2D arrays from the rows in the data frame.
    ###########################################################################
    col_z_map = np.full((ny, nx), np.nan)

    # Bin type
    if bin_type == "default":
        for rr in range(df_gal.shape[0]):
            if "x (pixels)" in df_gal and "y (pixels)" in df_gal:
                xx = int(df_gal.iloc[rr]["x (pixels)"])
                yy = int(df_gal.iloc[rr]["y (pixels)"])
            elif "x, y (pixels)" in df:
                xx, yy = [int(cc) for cc in df_gal.iloc[rr]["x, y (pixels)"]]
            else:
                raise ValueError
            col_z_map[yy, xx] = df_gal.iloc[rr][col_z]
    else:
        if survey == "sami":
            hdulist = fits.open(
                Path(settings["sami"]["input_path"]) / f"ifs/{gal}/{gal}_A_{bin_type}_blue.fits.gz")
            bin_map = hdulist[2].data.astype("float")
            bin_map[bin_map == 0] = np.nan
            for nn in df_gal["Bin number"]:
                bin_mask = bin_map == nn
                col_z_map[bin_mask] = df_gal.loc[df_gal["Bin number"] == nn, col_z]
        else:
            raise ValueError("Bin types other than 'default' for surveys other than SAMI have not yet been implemented!")
            #TODO perhaps make bin_map one of the input arguments?

    ###########################################################################
    # Create the WCS for the axes
    ###########################################################################
    if plot_ra_dec:
        # Get the RA and Dec of the target
        if ra_deg is None and dec_deg is None and "RA (J2000)" in df_gal and "Dec (J2000)" in df:
            ra_deg = df_gal["RA (J2000)"].values[0]
            dec_deg = df_gal["Dec (J2000)"].values[0]
        else:
            raise ValueError("If plot_ra_dec is True, ra_deg and dec_deg must be specified if 'RA (J2000)' or 'Dec (J2000)' are not in df!")
        # Construct the WCS
        wcs = WCS(naxis=2)
        wcs.wcs.crval = [ra_deg, dec_deg]
        wcs.wcs.crpix = [x0_px, y0_px]
        wcs.wcs.cdelt = [-as_per_px / 3600, as_per_px / 3600]
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        
    else:
        # This code from https://docs.astropy.org/en/stable/visualization/wcsaxes/generic_transforms.html  
        # Set up an affine transformation
        transform = Affine2D()
        transform.scale(as_per_px)
        transform.translate(- x0_px * as_per_px, - y0_px * as_per_px)

        # Set up metadata dictionary
        coord_meta = {}
        coord_meta["name"] = "x", "y"
        coord_meta["type"] = "scalar", "scalar"
        coord_meta["wrap"] = None, None
        coord_meta["unit"] = u.arcsec, u.arcsec
        coord_meta["format_unit"] = u.arcsec, u.arcsec

    ###########################################################################
    # Plot
    ###########################################################################
    # If no axis is specified then create a new one with a vertical colorbar.
    if ax is None:
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, subplot_kw={"projection": wcs})
        fig = plt.figure(figsize=figsize)
        if plot_ra_dec:
            ax = fig.add_axes([0.1, 0.1, 0.6, 0.8], projection=wcs)
        else:
            ax = WCSAxes(fig, [0.1, 0.1, 0.8, 0.8], aspect="equal", transform=transform, coord_meta=coord_meta)
            fig.add_axes(ax)
    else:
        # Sneaky... replace the provided axis with one that has the correct projection
        fig = ax.get_figure()
        bbox = ax.get_position()
        ax.remove()
        if plot_ra_dec:
            ax = fig.add_axes(bbox, projection=wcs)
        else:
            ax = WCSAxes(fig, bbox, aspect="equal", transform=transform, coord_meta=coord_meta)
            fig.add_axes(ax)

    # Minimum data range
    if vmin is None:
        vmin = get_vmin(col_z)
    elif vmin == "auto":
        vmin = np.nanmin(col_z_map)

    # Maximum data range
    if vmax is None:
        vmax = get_vmax(col_z)
    elif vmax == "auto":
        vmax = np.nanmax(col_z_map)

    # options for cmap are None --> use default cmap; str --> use that cmap
    discrete_colourmap = False
    if cmap is None:
        res = get_cmap(col_z)
        if type(res) == tuple:
            cmap, cax_ticks, cax_labels = res
            discrete_colourmap = True
        else:
            cmap = res
    elif type(cmap) == str:
        cmap = plt.cm.get_cmap(cmap).copy()
    cmap.set_bad("#b3b3b3")
    m = ax.imshow(col_z_map, cmap=cmap, vmin=vmin, vmax=vmax)

    ###########################################################################
    # Contours
    ###########################################################################
    if contours:
        if col_z_contours not in df_gal:
            raise ValueError(f"{col_z_contours} not found in df_gal!")
        
        # Create empty array to store contour values
        col_z_contour_map = np.full((ny, nx), np.nan)        
        if survey == "sami" and (bin_type == "adaptive" or bin_type == "sectors"):
            hdulist = fits.open(
                Path(settings["sami"]["input_path"]) / f"ifs/{gal}/{gal}_A_{bin_type}_blue.fits.gz")
            bin_map = hdulist[2].data.astype("float")
            bin_map[bin_map == 0] = np.nan
            for nn in df_gal["Bin number"]:
                bin_mask = bin_map == nn
                col_z_contour_map[bin_mask] = df_gal.loc[
                    df_gal["Bin number"] == nn, col_z_contours]
        elif bin_type == "default":
            for rr in range(df_gal.shape[0]):
                if "x (pixels)" in df_gal and "y (pixels)" in df_gal:
                    xx = int(df_gal.iloc[rr]["x (pixels)"])
                    yy = int(df_gal.iloc[rr]["y (pixels)"])
                elif "x, y (pixels)" in df_gal:
                    xx, yy = [int(cc) for cc in df_gal.iloc[rr]["x, y (pixels)"]]
                col_z_contour_map[yy, xx] = df_gal.iloc[rr][col_z_contours]

        # Draw contours
        ax.contour(col_z_contour_map,
                    linewidths=linewidths,
                    colors=colors,
                    levels=10 if levels is None else levels)

    ###########################################################################
    # Colourbar
    ###########################################################################
    # If plot_colorbar is True but the colorbar axis is not specified,
    # then create a new one.
    if plot_colorbar and cax is None:
        bbox = ax.get_position()
        if cax_orientation == "vertical":
            cax = fig.add_axes(
                [bbox.x0 + bbox.width, bbox.y0, bbox.width * 0.1, bbox.height])
        elif cax_orientation == "horizontal":
            cax = fig.add_axes([
                bbox.x0, bbox.y0 + bbox.height, bbox.width, bbox.height * 0.1
            ])

    # Add labels & ticks to colourbar, if necessary
    if plot_colorbar:
        plt.colorbar(mappable=m, cax=cax, orientation=cax_orientation)
        if cax_orientation == "vertical":
            cax.set_ylabel(get_label(col_z))
        elif cax_orientation == "horizontal":
            cax.xaxis.set_ticks_position("top")
            cax.xaxis.set_label_position('top')
            cax.set_xlabel(get_label(col_z))

        if discrete_colourmap:
            if cax_orientation == "vertical":
                cax.yaxis.set_ticks(cax_ticks)
                cax.yaxis.set_ticklabels(cax_labels)
            elif cax_orientation == "horizontal":
                cax.xaxis.set_ticks(cax_ticks)
                cax.xaxis.set_ticklabels(cax_labels)

    ###########################################################################
    # Decorations
    ###########################################################################
    # Include scale bar
    if show_scale_bar and "kpc per arcsec" in df_gal:
        plot_scale_bar(as_per_px=as_per_px,
                       kpc_per_as=df_gal["kpc per arcsec"].unique()[0],
                       fontsize=10,
                       ax=ax,
                       l=10,
                       units="arcsec",
                       color="black",
                       loffset=0.30,
                       long_dist_str=False)
    if show_compass:
        plot_compass(ax=ax, color="black", PA_deg=PA_deg)

    # Title
    if show_title:
       ax.set_title(str(gal))

    # Axis labels
    if axis_labels and plot_ra_dec:
        ax.set_ylabel("Dec (J2000)")
        ax.set_xlabel("RA (J2000)")

    return fig, ax
