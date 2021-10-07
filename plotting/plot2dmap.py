import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from plotting.plottools import vmin_fn, vmax_fn, label_fn, cmap_fn, plot_scale_bar, plot_compass

import matplotlib.pyplot as plt
plt.ion()

sami_datacube_path = "/priv/myrtle1/sami/sami_data/Final_SAMI_data/cube/sami/dr3/"

###############################################################################
def plot2dmap(df_gal, col_z, bin_type,
              show_title=True, axis_labels=True,
              vmin=None, vmax=None,
              ax=None, plot_colorbar=True, cax=None, cax_orientation="vertical",
              figsize=(5, 5)):
    """
    Show a 2D map of the galaxy, where sectors/bins/spaxels are coloured by
    the desired quantities.
    """
    assert col_z in df_gal.columns,\
        f"{col_z} is not a valid column!"
    assert cax_orientation == "horizontal" or cax_orientation == "vertical",\
        "cax_orientation must be either 'horizontal' or 'vertical'!"
    assert len(df_gal.catid.unique()) == 1,\
        "df_gal contains must only contain one galaxy!"

    # Load the non-binned data cube to get a continuum image
    gal = df_gal.catid.unique()[0]
    hdulist = fits.open(os.path.join(sami_datacube_path, f"ifs/{gal}/{gal}_A_cube_blue.fits.gz"))
    data_cube = hdulist[0].data
    header = hdulist[0].header
    lambda_0_A = header["CRVAL3"] - header["CRPIX3"] * header["CDELT3"]
    dlambda_A = header["CDELT3"]
    N_lambda = header["NAXIS3"]
    lambda_vals_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A 
    start_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + df_gal["z_spec"].unique()[0]) - 4000))
    stop_idx = np.nanargmin(np.abs(lambda_vals_A / (1 + df_gal["z_spec"].unique()[0]) - 5000))
    im_B = np.nansum(data_cube[start_idx:stop_idx], axis=0)
    im_B[im_B == 0] = np.nan

    # Also, get the WCS.
    wcs = WCS(hdulist[0].header).dropaxis(2)  # ?? check this
    hdulist.close()

    # Reconstruct 2D arrays from the rows in the data frame.
    col_z_map = np.full((50, 50), np.nan)
    if bin_type == "adaptive":
        hdulist = fits.open(os.path.join(sami_data_path, f"ifs/{gal}/{gal}_A_{bin_type}_blue.fits.gz"))
        bin_map = hdulist[2].data.astype("float")
        bin_map[bin_map==0] = np.nan
        for ii in df_gal["bin_number"]:
            bin_mask = bin_map == ii
            col_z_map[bin_mask] = df_gal.loc[df_gal["bin_number"] == ii, col_z]

    elif bin_type == "default":
        df_gal["x, y (pixels)"] = list(zip(df_gal["x (projected, arcsec)"] * 2, df_gal["y (projected, arcsec)"] * 2))
        for rr in range(df_gal.shape[0]):
            xx, yy = [int(cc) for cc in df_gal.iloc[rr]["x, y (pixels)"]]
            col_z_map[yy, xx] = df_gal.iloc[rr][col_z]

    # If no axis is specified then create a new one with a vertical colorbar.
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, subplot_kw={"projection": wcs})
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
    cmap = cmap_fn(col_z)
    cmap.set_bad("#b3b3b3")
    m = ax.imshow(col_z_map, cmap=cmap, vmin=vmin, vmax=vmax)

    # Contours
    ax.contour(im_B, linewidths=0.5, colors="white", levels=np.logspace(0, 2.5, 15))

    # Include scale bar
    plot_scale_bar(as_per_px=np.abs(header["CDELT1"]) * 3600, kpc_per_as=df_gal["kpc per arcsec"].unique()[0], ax=ax, l=10, units="arcsec", color="black", loffset=0.30)
    plot_compass(ax=ax, color="black")

    if show_title:
        ax.set_title(f"GAMA{gal}")

    # If the user wants to plot a colorbar but the colorbar axis is not specified,
    # then create a new one.
    if plot_colorbar and cax is None:
        bbox = ax.get_position()
        # Shrink axis first
        if cax_orientation == "vertical":
            ax.set_position([bbox.x0, bbox.y0, bbox.width * .85, bbox.height])
            cax = fig.add_axes([bbox.x0 + bbox.width * .85, bbox.y0, 0.05, bbox.height])
        elif cax_orientation == "horizontal":
            ax.set_position([bbox.x0, bbox.y0, bbox.width, bbox.height * 0.85])
            cax = fig.add_axes([bbox.x0, bbox.y0 + bbox.height * 0.85, bbox.width, 0.05])

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

    # Axis labels
    if axis_labels:
        ax.set_ylabel("Dec (J2000)")
        ax.set_xlabel("RA (J2000)")

    return fig


