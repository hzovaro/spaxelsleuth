# Imports
import pandas as pd
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

from spaxelsleuth.plotting.plottools import get_cmap, get_vmin, get_vmax, get_label, trim_suffix

import logging
logger = logging.getLogger(__name__)

###############################################################################
def plot2dhist(df, col_x, col_y, col_z, ax, log_z, vmin, vmax, xmin, xmax,
               ymin, ymax, nbins, alpha, cmap):
    """
    Plot a 2D histogram corresponding to col_x and col_y in DataFrame df,
    optionally coloured by the median value of a third parameter col_z in 
    each histogram cell. 

    INPUTS
    --------------------------------------------------------------------------
    df:                 pandas DataFrame
        DataFrame that has been created using make_df_sami.py or has a similar
        format.
    
    col_x:              str
        X-coordinate quantity. Must be a column in df. col_x can correspond to
        a specific column (e.g. "sigma_gas (component 1)"); alternatively,
        it can be left unspecified (i.e. "sigma_gas") in which case data in 
        all components (i.e., component 1, 2 and 3 if ncomponnets == "recom")
        is plotted together.
    
    col_y:              str
        Y-coordinate quantity. Must be a column in df. col_y can correspond to
        a specific column (e.g. "sigma_gas (component 1)"); alternatively,
        it can be left unspecified (i.e. "sigma_gas") in which case data in 
        all components (i.e., component 1, 2 and 3 if ncomponnets == "recom")
        is plotted together.
    
    col_z:              str
        Quantity used to colour the histogram. Must be a column in df or "count".
        NOTE: if you want to plot discrete quantities, such as BPT category,
        then you must specify the numeric option for these, i.e. set 
        col_z = "BPT (numeric)" rather than "BPT".

    ax:                 matplotlib.axis 
        Axis on which to plot.

    log_z:              bool
        Whether to scale the z-axis colour of the histogram logarithmically.
    
    vmin:               float
        Minimum value to use for marker colour if col_z is set.
    
    vmax:               float
        Maximum value to use for marker colour if col_z is set.
    
    xmin:               float
        Minimum x-axis value. Defaults to vmin_fn(col_x) in plottools.py.
    
    xmax:               float
        Maximum x-axis value. Defaults to vmax_fn(col_x) in plottools.py.
    
    ymin:               float
        Minimum y-axis value. Defaults to vmin_fn(col_y) in plottools.py.
    
    ymax:               float
        Maximum y-axis value. Defaults to vmax_fn(col_y) in plottools.py.
    
    nbins:              int
        Number of bins in x and y to use when drawing the 2D histogram.
    
    alpha:              float
        Transparency of histogram.
    
    cmap:               str or Matplotlib colourmap instance
        Colourmap used to plot the z-axis quantity. Defaults to cmap_fn(col_y) 
        in plottools.py.

    OUTPUTS
    ---------------------------------------------------------------------------
    The "mappable" object returned by histhelper() that can be passed to 
    plt.colorbar() to create a colourbar.

    """

    # Figure out how many components were fitted.
    ncomponents = "recom" if any(
        [c.endswith("(component 3)") for c in df.columns]) else "1"

    # If either column are present as multiple components, then make a new
    # data frame containing all of them.
    if f"{col_x} (component 1)" in df.columns:
        if f"{col_y} (component 1)" in df.columns:
            data_x = np.concatenate((df[f"{col_x} (component 1)"].values,
                                     df[f"{col_x} (component 2)"].values,
                                     df[f"{col_x} (component 3)"].values
                                     )) if ncomponents == "recom" else df[
                                         f"{col_x} (component 1)"].values
            data_y = np.concatenate((df[f"{col_y} (component 1)"].values,
                                     df[f"{col_y} (component 2)"].values,
                                     df[f"{col_y} (component 3)"].values
                                     )) if ncomponents == "recom" else df[
                                         f"{col_y} (component 1)"].values
        else:
            data_x = df[
                f"{col_x} (total)"] if f"{col_x} (total)" in df.columns else df[
                    f"{col_x}"]
            data_y = df[col_y].values
    else:
        if f"{col_y} (component 1)" in df.columns:
            data_y = df[
                f"{col_y} (total)"] if f"{col_y} (total)" in df.columns else df[
                    f"{col_y}"]
            data_x = df[col_x].values
        else:
            data_x = df[col_x].values
            data_y = df[col_y].values

    if col_z == "count":
        df_sub = pd.DataFrame({col_x: data_x, col_y: data_y})
    else:
        # If col_z has individual measurements for each component...
        if f"{col_z} (component 1)" in df.columns:
            # If x and y also have individual measurements for each component, then use all 3 for x, y and z.
            if f"{col_x} (component 1)" in df.columns and f"{col_y} (component 1)" in df.columns:
                data_z = np.concatenate((df[f"{col_z} (component 1)"].values,
                                         df[f"{col_z} (component 2)"].values,
                                         df[f"{col_z} (component 3)"].values
                                         )) if ncomponents == "recom" else df[
                                             f"{col_z} (component 1)"].values
            # Otherwise, just use the "total" measurement.
            else:
                data_z = df[
                    f"{col_z} (total)"] if f"{col_z} (total)" in df.columns else df[
                        f"{col_z}"]
        # Otherwise, just use the column as-is.
        else:
            # If x and y are measured for each component, but z isn't, then repeat the z data for each component
            if f"{col_x} (component 1)" in df.columns and f"{col_y} (component 1)" in df.columns:
                data_z = np.concatenate(
                    (df[col_z].values, df[col_z].values, df[col_z].values
                     )) if ncomponents == "recom" else df[col_z].values
            else:
                data_z = df[col_z]
        df_sub = pd.DataFrame({col_x: data_x, col_y: data_y, col_z: data_z})

    def _histhelper(df):
        """Helper function used to plot 2D histograms."""
        # Determine bin edges for the x & y-axis line ratio
        # Messy hack to include that final bin...
        ybins = np.linspace(ymin, ymax, nbins)
        dy = np.diff(ybins)[0]
        ybins = list(ybins)
        ybins.append(ybins[-1] + dy)
        ybins = np.array(ybins)
        ycut = pd.cut(df[col_y], ybins)

        xbins = np.linspace(xmin, xmax, nbins)
        dx = np.diff(xbins)[0]
        xbins = list(xbins)
        xbins.append(xbins[-1] + dx)
        xbins = np.array(xbins)
        xcut = pd.cut(df[col_x], xbins)

        # Combine the x- and y-cuts
        cuts = pd.DataFrame({"xbin": xcut, "ybin": ycut})

        # Function for colouring the histogram: if it's a continuous property, e.g.
        # SFR, then use the median. If it's a discrete quantity, e.g. BPT category,
        # then use the mode (= most frequent number in a data set). This will
        # help to avoid the issue in which np.nanmedian returns a non-integer value.
        if col_z.startswith("BPT") or col_z.startswith(
                "Morphology") or col_z.startswith("WHAV*"):

            def mode(data):
                vals, counts = np.unique(data, return_counts=True)
                idx = np.nanargmax(counts)
                return vals[idx]

            func = mode
        else:
            func = np.nanmedian

        # Calculate the desired quantities for the data binned by x and y
        gb_binned = df.join(cuts).groupby(list(cuts))
        if col_z == "count":
            df_binned = gb_binned.agg({df.columns[0]: lambda g: g.count()})
            df_binned = df_binned.rename(columns={df.columns[0]: "count"})
        else:
            df_binned = gb_binned.agg({col_z: func})

        # Pull out arrays to plot
        count_map = df_binned[col_z].values.reshape((nbins, nbins))

        # Plot.
        if log_z:
            m = ax.pcolormesh(xbins[:-1],
                              ybins[:-1],
                              count_map.T,
                              cmap=cmap,
                              edgecolors="none",
                              shading="auto",
                              norm=LogNorm(vmin=vmin, vmax=vmax))
        else:
            m = ax.pcolormesh(xbins[:-1],
                              ybins[:-1],
                              count_map.T,
                              cmap=cmap,
                              edgecolors="none",
                              vmin=vmin,
                              vmax=vmax,
                              shading="auto")
        m.set_rasterized(True)

        # Dodgy...
        if alpha < 1:
            overlay = np.full_like(count_map.T, 1.0)
            mo = ax.pcolormesh(xbins[:-1],
                               ybins[:-1],
                               overlay,
                               alpha=1 - alpha,
                               cmap="gray",
                               vmin=0,
                               vmax=1,
                               shading="auto")
            mo.set_rasterized(True)

        return m

    # If we're plotting the BPT categories, also want to show the "uncategorised" ones.
    if col_z.startswith("BPT (numeric)") or col_z.startswith(
            "WHAV* (numeric)"):
        df_classified = df[df[col_z] > -1]
        df_unclassified = df[df[col_z] == -1]
        if df_unclassified.shape[0] > 0 and np.any(
                ~np.isnan(df_unclassified[col_x])) and np.any(
                    ~np.isnan(df_unclassified[col_y])):
            _histhelper(df=df_unclassified)
        m = _histhelper(df=df_classified)
    else:
        m = _histhelper(df=df_sub)

    return m


###############################################################################
def plot2dcontours(df, col_x, col_y, ax, nbins, alpha, levels, xmin, xmax,
                   ymin, ymax, linewidths, colors):
    """
    Plot a 2D histogram corresponding to col_x and col_y in DataFrame df,
    optionally coloured by the median value of a third parameter col_z in 
    each histogram cell. 

    INPUTS
    --------------------------------------------------------------------------
    df:                 pandas DataFrame
        DataFrame that has been created using make_df_sami.py or has a similar
        format.
    
    col_x:              str
        X-coordinate quantity. Must be a column in df. col_x can correspond to
        a specific column (e.g. "sigma_gas (component 1)"); alternatively,
        it can be left unspecified (i.e. "sigma_gas") in which case data in 
        all components (i.e., component 1, 2 and 3 if ncomponnets == "recom")
        is plotted together.
    
    col_y:              str
        Y-coordinate quantity. Must be a column in df. col_y can correspond to
        a specific column (e.g. "sigma_gas (component 1)"); alternatively,
        it can be left unspecified (i.e. "sigma_gas") in which case data in 
        all components (i.e., component 1, 2 and 3 if ncomponnets == "recom")
        is plotted together.

    ax:                 matplotlib.axis 
        Axis on which to plot.

    nbins:              int
        Number of bins in x and y to use when drawing the 2D histogram on which
        the contours are based.

    alpha:              float
        Contour transparency.
    
    xmin:               float
        Minimum x-axis value. Defaults to vmin_fn(col_x) in plottools.py.
    
    xmax:               float
        Maximum x-axis value. Defaults to vmax_fn(col_x) in plottools.py.
    
    ymin:               float
        Minimum y-axis value. Defaults to vmin_fn(col_y) in plottools.py.
    
    ymax:               float
        Maximum y-axis value. Defaults to vmax_fn(col_y) in plottools.py.
    
    linewidths:         float
        Contour linewidths.

    colors:             str
        Contour colours.

    OUTPUTS
    ---------------------------------------------------------------------------
    The "mappable" object returned by plt.contour() that can be passed to 
    plt.colorbar() to create a colourbar.

    """
    # Figure out how many components were fitted.
    ncomponents = "recom" if any(
        [c.endswith("(component 3)") for c in df.columns]) else "1"

    # If either column are present as multiple components, then make a new
    # data frame containing all of them.
    if f"{col_x} (component 1)" in df.columns:
        if f"{col_y} (component 1)" in df.columns:
            data_x = np.concatenate((df[f"{col_x} (component 1)"].values,
                                     df[f"{col_x} (component 2)"].values,
                                     df[f"{col_x} (component 3)"].values
                                     )) if ncomponents == "recom" else df[
                                         f"{col_x} (component 1)"].values
            data_y = np.concatenate((df[f"{col_y} (component 1)"].values,
                                     df[f"{col_y} (component 2)"].values,
                                     df[f"{col_y} (component 3)"].values
                                     )) if ncomponents == "recom" else df[
                                         f"{col_y} (component 1)"].values
        else:
            data_x = df[
                f"{col_x} (total)"] if f"{col_x} (total)" in df.columns else df[
                    f"{col_x}"]
            data_y = df[col_y].values
    else:
        if f"{col_y} (component 1)" in df.columns:
            data_y = df[
                f"{col_y} (total)"] if f"{col_y} (total)" in df.columns else df[
                    f"{col_y}"]
            data_x = df[col_x].values
        else:
            data_x = df[col_x].values
            data_y = df[col_y].values
    df = pd.DataFrame({col_x: data_x, col_y: data_y})

    # Determine bin edges for the x & y-axis line ratio
    # Messy hack to include that final bin...
    ybins = np.linspace(ymin, ymax, nbins)
    dy = np.diff(ybins)[0]
    ybins = list(ybins)
    ybins.append(ybins[-1] + dy)
    ybins = np.array(ybins)
    ycut = pd.cut(df[col_y], ybins)

    xbins = np.linspace(xmin, xmax, nbins)
    dx = np.diff(xbins)[0]
    xbins = list(xbins)
    xbins.append(xbins[-1] + dx)
    xbins = np.array(xbins)
    xcut = pd.cut(df[col_x], xbins)

    # Combine the x- and y-cuts
    cuts = pd.DataFrame({
        "xbin": xcut,
        "ybin": ycut
    })  # Combine the x- and y-cuts
    cuts = pd.DataFrame({"xbin": xcut, "ybin": ycut})

    # Calculate the desired quantities for the data binned by x and y
    gb_binned = df.join(cuts).groupby(list(cuts))
    df_binned = gb_binned.agg({df.columns[0]: lambda g: g.count()})

    # Pull out arrays to plot
    count_map = df_binned[df.columns[0]].values.reshape((nbins, nbins))

    # Plot.
    m = ax.contour(xbins[:-1] + dx / 2,
                   ybins[:-1] + dy / 2,
                   count_map.T,
                   levels=levels,
                   colors=colors,
                   alpha=alpha,
                   linewidths=linewidths)

    return m


###############################################################################
def plot2dhistcontours(df,
                       col_x,
                       col_y,
                       col_z=None,
                       log_z=False,
                       vmin=None,
                       vmax=None,
                       xmin=None,
                       xmax=None,
                       ymin=None,
                       ymax=None,
                       nbins=100,
                       ax=None,
                       axis_labels=True,
                       cmap=None,
                       plot_colorbar=True,
                       cax=None,
                       cax_orientation="vertical",
                       alpha=1.0,
                       hist=True,
                       contours=True,
                       levels=None,
                       linewidths=0.5,
                       colors="k",
                       figsize=(9, 6)):
    """
    Plot a 2D histogram of the data in columns col_x and col_y in a pandas 
    DataFrame df. Optionally, overlay contours showing the corresponding 
    number distribution.

    INPUTS
    --------------------------------------------------------------------------
    df:                 pandas DataFrame
        DataFrame that has been created using make_df_sami.py or has a similar
        format.
    
    col_x:              str
        X-coordinate quantity. Must be a column in df. col_x can correspond to
        a specific column (e.g. "sigma_gas (component 1)"); alternatively,
        it can be left unspecified (i.e. "sigma_gas") in which case data in 
        all components (i.e., component 1, 2 and 3 if ncomponnets == "recom")
        is plotted together.
    
    col_y:              str
        Y-coordinate quantity. Must be a column in df. col_y can correspond to
        a specific column (e.g. "sigma_gas (component 1)"); alternatively,
        it can be left unspecified (i.e. "sigma_gas") in which case data in 
        all components (i.e., component 1, 2 and 3 if ncomponnets == "recom")
        is plotted together.
    
    col_z:              str
        Quantity used to colour the histogram. Must be a column in df. col_z 
        can correspond to a specific column (e.g. "sigma_gas (component 1)"); 
        alternatively, it can be left unspecified (i.e. "sigma_gas") in which 
        case data in all components (i.e., component 1, 2 and 3 if 
        ncomponnets == "recom") is plotted together.
    
        NOTE: if you want to plot discrete quantities, such as BPT category,
        then you must specify the numeric option for these, i.e. set 
        col_z = "BPT (numeric)" rather than "BPT".

    log_z:              bool
        Whether to scale the z-axis colour of the histogram logarithmically.
    
    vmin:               float
        Minimum value to use for marker colour if col_z is set.
    
    vmax:               float
        Maximum value to use for marker colour if col_z is set.
    
    xmin:               float
        Minimum x-axis value. Defaults to vmin_fn(col_x) in plottools.py.
    
    xmax:               float
        Maximum x-axis value. Defaults to vmax_fn(col_x) in plottools.py.
    
    ymin:               float
        Minimum y-axis value. Defaults to vmin_fn(col_y) in plottools.py.
    
    ymax:               float
        Maximum y-axis value. Defaults to vmax_fn(col_y) in plottools.py.
    
    nbins:              int
        Number of bins in x and y to use when drawing the 2D histogram.

    ax:                 matplotlib.axis 
        Axis on which to plot. If unspecified, a new figure is created.    
    
    axis_labels:        bool
        Whether to apply axis labels as returned by get_label(col_<x/y>) in 
        plottools.py.
    
    cmap:               str
        Matplotlib colourmap to use. Defaults cmap_fn(col_z) in plottools.py.
    
    plot_colorbar:      bool
        Whether to plot a colourbar.
    
    cax:                matplotlib.axis
        Axis in which to plot colourbar if plot_colorbar is True. If no axis 
        is specified, a new colourbar axis is created to the side of the 
        main figure axis.
    
    cax_orientation:    str
        Colourbar orientation. May be "vertical" (default) or "horizontal".
    
    alpha:              float
        Transparency of histogram.
    
    zorder:             int
        Z-order of histogram.
    
    hist:               bool
        If True, plot the 2D histogram.

    contours:           bool
        If True, overlay contours showing the density distribution in the 
        histrogram. 

    levels:             Numpy array
        Contour levels.

    linewidths:         float 
        Contour linewidths.

    colors:             str
        Contour colours.
    
    figsize:            tuple (width, height)
        Figure size in inches.

    ---------------------------------------------------------------------------
    Returns:
        matplotlib figure object that is the parent of the main axis.
    """
    if col_z is None and hist is True:
        raise ValueError("if hist is True then col_z must be specified!")
    else:
        if col_z != "count" and df[col_z].dtype == "O":
            col, suffix = trim_suffix(col_z)
            if f"{col} (numeric)" + suffix in df:
                col_z = f"{col} (numeric)" + suffix
            else:
                raise ValueError(
                    f"{col_z} has an object data type and no numeric counterpart exists in df!"
                )
    if cax_orientation not in ["horizontal", "vertical"]:
        raise ValueError(
            "cax_orientation must be either 'horizontal' or 'vertical'!")

    # If no axis is specified then create a new one with a vertical colorbar.
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])
    else:
        fig = ax.get_figure()

    # If the user wants to plot a colorbar but the colorbar axis is not specified, then create a new one.
    if plot_colorbar and cax is None:
        bbox = ax.get_position()
        # Shrink axis first
        if cax_orientation == "vertical":
            cax = fig.add_axes(
                [bbox.x0 + bbox.width, bbox.y0, bbox.width * 0.1, bbox.height])
        elif cax_orientation == "horizontal":
            cax = fig.add_axes([
                bbox.x0, bbox.y0 + bbox.height, bbox.width, bbox.height * 0.1
            ])

    # Minimum/maximum data ranges
    if vmin is None:
        vmin = get_vmin(col_z)
    elif vmin == "auto":
        vmin = df[col_z].min()
    if vmax is None:
        vmax = get_vmax(col_z)
    elif vmax == "auto":
        vmax = df[col_z].max()
    if xmin is None:
        xmin = get_vmin(col_x)
    elif xmin == "auto":
        xmin = df[col_x].min()
    if xmax is None:
        xmax = get_vmax(col_x)
    elif xmax == "auto":
        xmax = df[col_x].max()
    if ymin is None:
        ymin = get_vmin(col_y)
    elif ymin == "auto":
        ymin = df[col_y].min()
    if ymax is None:
        ymax = get_vmax(col_y)
    elif ymax == "auto":
        ymax = df[col_y].max()

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
    cmap.set_bad("white", alpha=0)

    # Plot histogram and/or contours
    if hist:
        m = plot2dhist(df=df,
                       col_x=col_x,
                       col_y=col_y,
                       col_z=col_z,
                       log_z=log_z,
                       nbins=nbins,
                       ax=ax,
                       alpha=alpha,
                       cmap=cmap,
                       vmin=vmin,
                       vmax=vmax,
                       xmin=xmin,
                       xmax=xmax,
                       ymin=ymin,
                       ymax=ymax)
    if contours:
        plot2dcontours(df=df,
                       col_x=col_x,
                       col_y=col_y,
                       ax=ax,
                       alpha=alpha,
                       nbins=nbins,
                       linewidths=linewidths,
                       colors=colors,
                       levels=levels,
                       xmin=xmin,
                       xmax=xmax,
                       ymin=ymin,
                       ymax=ymax)

    # Colourbar
    if plot_colorbar and hist:
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

    # Axis labels
    if axis_labels:
        ax.set_xlabel(get_label(col_x))
        ax.set_ylabel(get_label(col_y))

    #////////////////////////////////////////////////
    # Demarcation lines
    #////////////////////////////////////////////////
    # WHAN diagram
    if col_y.startswith("log HALPHA EW"):
        # Classification lines of Lacerda+2017
        ax.axhline(np.log10(3), linestyle="--", linewidth=1, color="k")
        ax.axhline(np.log10(14), linestyle="--", linewidth=1, color="k")
        # Classification lines of Cid Fernandes+2011
        ax.axhline(np.log10(0.5), linestyle=":", linewidth=1,
                   color="k")  # "Passive" galaxies
        if col_x.startswith("log N2"):
            ax.plot([-0.4, get_vmax(col_x)],
                    [np.log10(6), np.log10(6)],
                    linestyle="-",
                    linewidth=1,
                    color="k"
                    )  # Optimal K06 dividing line between LINERs and Seyferts
            ax.axvline(
                -0.4, linestyle="-", linewidth=1, color="k"
            )  # Optimal K06 dividing line between SF and other mechanisms
        else:
            ax.axhline(np.log10(6), linestyle="-", linewidth=1,
                       color="k")  # Seyfert vs. LINER
    elif col_x.startswith("log HALPHA EW"):
        # Classification lines of Lacerda+2017
        ax.axvline(np.log10(3), linestyle="--", linewidth=1, color="k")
        ax.axvline(np.log10(14), linestyle="--", linewidth=1, color="k")
        # Classification lines of Cid Fernandes+2011
        ax.axvline(np.log10(0.5), linestyle=":", linewidth=1,
                   color="k")  # "Passive" galaxies
        if col_y.startswith("log N2"):
            ax.plot([np.log10(6), np.log10(6)], [-0.4, get_vmax(col_y)],
                    linestyle="-",
                    linewidth=1,
                    color="k"
                    )  # Optimal K06 dividing line between LINERs and Seyferts
            ax.axhline(
                -0.4, linestyle="-", linewidth=1, color="k"
            )  # Optimal K06 dividing line between SF and other mechanisms
        else:
            ax.axvline(np.log10(6), linestyle="-", linewidth=1,
                       color="k")  # Seyfert vs. LINER

    # Vertical/horizontal lines at 0
    elif col_x.startswith("sigma_gas - sigma_*"):
        ax.axvline(0, linestyle="-", linewidth=1, color="k")
    elif col_y.startswith("sigma_gas - sigma_*"):
        ax.axhline(0, linestyle="-", linewidth=1, color="k")

    # Re-set axis limits
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    return fig


###############################################################################
def plot2dscatter(df,
                  col_x,
                  col_y,
                  gal=None,
                  col_z=None,
                  vmin=None,
                  vmax=None,
                  xmin=None,
                  xmax=None,
                  ymin=None,
                  ymax=None,
                  ax=None,
                  axis_labels=True,
                  cmap=None,
                  plot_colorbar=True,
                  cax=None,
                  cax_orientation="vertical",
                  alpha=1.0,
                  zorder=2,
                  errorbars=True,
                  markeredgecolor="k",
                  markerfacecolor="k",
                  marker="o",
                  markersize=20,
                  figsize=(9, 7)):
    """
    Make a scatter plot comprising spaxels or individual line components stored 
    in a given Pandas DataFrame.

    INPUTS
    ---------------------------------------------------------------------------
    df:                 pandas DataFrame
        DataFrame that has been created using make_df_sami.py or has a similar
        format.
    
    col_x:              str
        X-coordinate quantity. Must be a column in df.
    
    col_y:              str
        Y-coordinate quantity. Must be a column in df.

    gal:                int 
        Galaxy (or galaxies) to plot, if desired. By default, all rows of df
        are plotted.
    
    col_z:              str
        Quantity used to colour the points. Must be a column in df. If not 
        specified, the points are all given the same colour specified by 
        markerfacecolor. 
        NOTE: if you want to plot discrete quantities, such as BPT category,
        then you must specify the numeric option for these, i.e. set 
        col_z = "BPT (numeric)" rather than "BPT".
    
    vmin:               float
        Minimum value to use for marker colour if col_z is set. Defaults to 
        vmin_fn(col_z) in plottools.py.
    
    vmax:               float
        Maximum value to use for marker colour if col_z is set. Defaults to 
        vmax_fn(col_z) in plottools.py.
    
    xmin:               float
        Minimum x-axis value. Defaults to vmin_fn(col_x) in plottools.py.
    
    xmax:               float
        Maximum x-axis value. Defaults to vmax_fn(col_x) in plottools.py.
    
    ymin:               float
        Minimum y-axis value. Defaults to vmin_fn(col_y) in plottools.py.
    
    ymax:               float
        Maximum y-axis value. Defaults to vmax_fn(col_y) in plottools.py.
    
    ax:                 matplotlib.axis 
        Axis on which to plot. If unspecified, a new figure is created.    
    
    axis_labels:        bool
        Whether to apply axis labels as returned by get_label(col_<x/y>) in 
        plottools.py.
    
    cmap:               str
        Matplotlib colourmap to use. Defaults cmap_fn(col_z) in plottools.py.
    
    plot_colorbar:      bool
        Whether to plot a colourbar. Defaults to True if col_z is specified,
        otherwise False. 
    
    cax:                matplotlib.axis
        Axis in which to plot colourbar if plot_colorbar is True. If no axis 
        is specified, a new colourbar axis is created to the side of the 
        main figure axis.
    
    cax_orientation:    str
        Colourbar orientation. May be "vertical" (default) or "horizontal".
    
    alpha:              float
        Transparency of scatter points.
    
    zorder:             int
        Z-order of scatter points.
    
    errorbars:          bool
        If True, plot 1-sigma error bars associated with the x- and y-axis 
        quantities.
    
    markeredgecolor:    str
        Marker edge colour. Defaults to black.
    
    markerfacecolor:    str
        Marker face colour. Defaults to black.
    
    marker:             str
        Marker shape, e.g. 'o' or 'x'.
    
    markersize:         float
        Marker size in pt.
    
    figsize:            tuple (width, height)
        Figure size in inches.


    OUTPUTS
    ---------------------------------------------------------------------------
    matplotlib figure object that is the parent of the main axis.

    """
    if col_z == "count":
        raise ValueError(f"{col_z} cannot be 'count' in a scatter plot!")
    if col_z is not None:
        for col in [col_x, col_y, col_z]:
            if not ((col in df.columns) or
                    (f"{col} (component 1)" in df.columns) or
                    (f"{col} (total)" in df.columns)):
                raise ValueError(f"{col} is not a valid column!")
        if df[col_z].dtype == "O":
            col, suffix = trim_suffix(col_z)
            if f"{col} (numeric)" + suffix in df:
                col_z = f"{col} (numeric)" + suffix
            else:
                raise ValueError(
                    f"{col_z} has an object data type and no numeric counterpart exists in df!"
                )
    if cax_orientation not in ["horizontal", "vertical"]:
        raise ValueError(
            "cax_orientation must be either 'horizontal' or 'vertical'!")
    if col_z is None and plot_colorbar == True:
        logger.warn("not plotting colourbar because col_z is not specified!")

    # If a galaxy is specified (or galaxies), then plot only the galaxy
    if gal is not None:
        #TODO replace "type()" with "isinstance()" elsewhere
        if isinstance(gal, int) or isinstance(gal, np.integer) or isinstance(gal, str):
            df_plot = df[df["ID"] == gal]
        elif isinstance(gal, list):
            df_plot = df[df["ID"].isin(gal)]
        else:
            raise ValueError("gal is not an integer, string or list so I don't know what to do :(")
    else:
        df_plot = df

    # If no axis is specified then create a new one with a vertical colorbar.
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])
    else:
        fig = ax.get_figure()

    # If the user wants to plot a colorbar but the colorbar axis is not specified,
    # then create a new one.
    if plot_colorbar and cax is None and col_z is not None:
        bbox = ax.get_position()
        # Shrink axis first
        if cax_orientation == "vertical":
            cax = fig.add_axes(
                [bbox.x0 + bbox.width, bbox.y0, bbox.width * 0.1, bbox.height])
        elif cax_orientation == "horizontal":
            cax = fig.add_axes([
                bbox.x0, bbox.y0 + bbox.height, bbox.width, bbox.height * 0.1
            ])

    # Deal with annoying case where there are multiple components
    if "component" in col_x:
        component_str = " (component" + col_x.split("(component")[1]
        col_x_arg = col_x.split(component_str)[0]
        col_x_err_label = f"{col_x_arg} error{component_str}"
        col_x_err_lower_label = f"{col_x_arg} error (lower){component_str}"
        col_x_err_upper_label = f"{col_x_arg} error (upper){component_str}"
    elif "(total)" in col_x:
        col_x_arg = col_x.split(" (total)")[0]
        col_x_err_label = f"{col_x_arg} error (total)"
        col_x_err_lower_label = f"{col_x_arg} error (lower) (total)"
        col_x_err_upper_label = f"{col_x_arg} error (upper) (total)"
    else:
        component_str = ""
        col_x_err_label = f"{col_x} error"
        col_x_err_lower_label = f"{col_x} error (lower)"
        col_x_err_upper_label = f"{col_x} error (upper)"

    if "component" in col_y:
        component_str = " (component" + col_y.split("(component")[1]
        col_y_arg = col_y.split(component_str)[0]
        col_y_err_label = f"{col_y_arg} error{component_str}"
        col_y_err_lower_label = f"{col_y_arg} error (lower){component_str}"
        col_y_err_upper_label = f"{col_y_arg} error (upper){component_str}"
    elif "(total)" in col_y:
        col_y_arg = col_y.split(" (total)")[0]
        col_y_err_label = f"{col_y_arg} error (total)"
        col_y_err_lower_label = f"{col_y_arg} error (lower) (total)"
        col_y_err_upper_label = f"{col_y_arg} error (upper) (total)"
    else:
        component_str = ""
        col_y_err_label = f"{col_y} error"
        col_y_err_lower_label = f"{col_y} error (lower)"
        col_y_err_upper_label = f"{col_y} error (upper)"

    # x-axis: add errorbars, if they exist
    if errorbars:
        # Symmetric errors
        if col_x_err_label in df_plot:
            xerr = df_plot[col_x_err_label]
            ax.errorbar(x=df_plot[col_x],
                        y=df_plot[col_y],
                        xerr=df_plot[col_x_err_label],
                        ls="none",
                        mec="none",
                        ecolor="k",
                        elinewidth=0.5,
                        alpha=alpha / 2,
                        zorder=zorder)
        # Asymmetric errors
        elif col_x_err_lower_label in df_plot and col_x_err_upper_label in df_plot:
            # Need to deal with special case where there is only one data point to prevent weird numpy error...
            if len(df_plot[col_x_err_lower_label]) > 1:
                xerr = np.array([
                    df_plot[col_x_err_lower_label],
                    df_plot[col_x_err_lower_label]
                ])
            else:
                xerr = np.array([
                    df_plot[col_x_err_lower_label].values[0],
                    df_plot[col_x_err_lower_label].values[0]
                ])[:, None]
            ax.errorbar(x=df_plot[col_x],
                        y=df_plot[col_y],
                        xerr=xerr,
                        ls="none",
                        mec="none",
                        ecolor="k",
                        elinewidth=0.5,
                        alpha=alpha / 2,
                        zorder=zorder)

        # y-axis: add errorbars, if they exist
        if col_y_err_label in df_plot:
            ax.errorbar(x=df_plot[col_x],
                        y=df_plot[col_y],
                        yerr=df_plot[col_y_err_label],
                        ls="none",
                        mec="none",
                        ecolor="k",
                        elinewidth=0.5,
                        alpha=alpha / 2,
                        zorder=zorder)
        elif col_y_err_lower_label in df_plot and col_y_err_upper_label in df_plot:
            # Need to deal with special case where there is only one data point to prevent weird numpy error...
            if len(df_plot[col_y_err_lower_label]) > 1:
                yerr = np.array([
                    df_plot[col_y_err_lower_label],
                    df_plot[col_y_err_upper_label]
                ])
            else:
                yerr = np.array([
                    df_plot[col_y_err_lower_label].values[0],
                    df_plot[col_y_err_upper_label].values[0]
                ])[:, None]
            ax.errorbar(x=df_plot[col_x],
                        y=df_plot[col_y],
                        yerr=yerr,
                        ls="none",
                        mec="none",
                        ecolor="k",
                        elinewidth=0.5,
                        alpha=alpha / 2,
                        zorder=zorder)

    # Only colour the spaxels of the galaxy.
    if col_z is not None:
        if vmin is None:
            vmin = get_vmin(col_z)
        if vmax is None:
            vmax = get_vmax(col_z)
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
        cmap.set_bad("white", alpha=0)

    if xmin is None:
        xmin = get_vmin(col_x)
    if xmax is None:
        xmax = get_vmax(col_x)
    if ymin is None:
        ymin = get_vmin(col_y)
    if ymax is None:
        ymax = get_vmax(col_y)

    # Plot the scatter points
    if col_z is not None:
        m = ax.scatter(x=df_plot[col_x],
                       y=df_plot[col_y],
                       c=df_plot[col_z],
                       cmap=cmap,
                       vmin=vmin,
                       vmax=vmax,
                       marker=marker,
                       edgecolors=markeredgecolor,
                       s=markersize,
                       alpha=alpha,
                       zorder=zorder + 1)
    else:
        m = ax.scatter(x=df_plot[col_x],
                       y=df_plot[col_y],
                       c=markerfacecolor,
                       marker=marker,
                       edgecolors=markeredgecolor,
                       s=markersize,
                       alpha=alpha,
                       zorder=zorder + 1)

    # Nice axis limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if plot_colorbar and col_z is not None:
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

    if axis_labels:
        ax.set_xlabel(get_label(col_x))
        ax.set_ylabel(get_label(col_y))

    # Add the EW classification lines of Lacerda+2017.
    if col_y.startswith("log HALPHA EW"):
        # Classification lines of Lacerda+2017
        ax.axhline(np.log10(3), linestyle="--", linewidth=1, color="k")
        ax.axhline(np.log10(14), linestyle="--", linewidth=1, color="k")
        # Classification lines of Cid Fernandes+2011
        ax.axhline(np.log10(0.5), linestyle=":", linewidth=1,
                   color="k")  # "Passive" galaxies
        if col_x.startswith("log N2"):
            ax.plot([-0.4, get_vmax(col_x)],
                    [np.log10(6), np.log10(6)],
                    linestyle="-",
                    linewidth=1,
                    color="k"
                    )  # Optimal K06 dividing line between LINERs and Seyferts
            ax.axvline(
                -0.4, linestyle="-", linewidth=1, color="k"
            )  # Optimal K06 dividing line between SF and other mechanisms
        else:
            ax.axhline(np.log10(6), linestyle="-", linewidth=1,
                       color="k")  # "Passive" galaxies

    elif col_x.startswith("log HALPHA EW"):
        # Classification lines of Lacerda+2017
        ax.axvline(np.log10(3), linestyle="--", linewidth=1, color="k")
        ax.axvline(np.log10(14), linestyle="--", linewidth=1, color="k")
        # Classification lines of Cid Fernandes+2011
        ax.axvline(np.log10(0.5), linestyle=":", linewidth=1,
                   color="k")  # "Passive" galaxies
        if col_y.startswith("log N2"):
            ax.plot([np.log10(6), np.log10(6)], [-0.4, get_vmax(col_y)],
                    linestyle="-",
                    linewidth=1,
                    color="k"
                    )  # Optimal K06 dividing line between LINERs and Seyferts
            ax.axhline(
                -0.4, linestyle="-", linewidth=1, color="k"
            )  # Optimal K06 dividing line between SF and other mechanisms
        else:
            ax.axvline(np.log10(6), linestyle="-", linewidth=1,
                       color="k")  # "Passive" galaxies

    elif col_x.startswith("sigma_gas - sigma_*"):
        # Vertical line at 0
        ax.axvline(0, linestyle="-", linewidth=1, color="k")
    elif col_y.startswith("sigma_gas - sigma_*"):
        # Vertical line at 0
        ax.axhline(0, linestyle="-", linewidth=1, color="k")

    # Re-set axis limits
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    return fig
