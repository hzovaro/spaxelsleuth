import pandas as pd
import numpy as np

from spaxelsleuth.plotting.plottools import cmap_fn, vmin_fn, vmax_fn, label_fn, histhelper
from spaxelsleuth.plotting.plottools import bpt_ticks, bpt_labels, law2021_ticks, law2021_labels, morph_ticks, morph_labels, ncomponents_ticks, ncomponents_labels

import matplotlib.pyplot as plt
plt.ion()

from IPython.core.debugger import Tracer

###############################################################################
def plot2dhist(df, col_x, col_y, col_z, ax, log_z=False,
               vmin=None, vmax=None,
               xmin=None, xmax=None,
               ymin=None, ymax=None,
               nbins=100, alpha=1.0, cmap=None):

    # Figure out how many components were fitted.
    ncomponents = "recom" if any([c.endswith("(component 2)") for c in df.columns]) else "1"

    # If either column are present as multiple components, then make a new 
    # data frame containing all of them.
    if f"{col_x} (component 0)" in df.columns:
        if f"{col_y} (component 0)" in df.columns:
            data_x = np.concatenate((df[f"{col_x} (component 0)"].values, df[f"{col_x} (component 1)"].values, df[f"{col_x} (component 2)"].values)) if ncomponents == "recom" else df[f"{col_x} (component 0)"].values
            data_y = np.concatenate((df[f"{col_y} (component 0)"].values, df[f"{col_y} (component 1)"].values, df[f"{col_y} (component 2)"].values)) if ncomponents == "recom" else df[f"{col_y} (component 0)"].values
        else:
            data_x = df[f"{col_x} (total)"] if f"{col_x} (total)" in df.columns else df[f"{col_x}"]
            data_y = df[col_y].values
    else:
        if f"{col_y} (component 0)" in df.columns:
            data_y = df[f"{col_y} (total)"] if f"{col_y} (total)" in df.columns else df[f"{col_y}"]
            data_x = df[col_x].values
        else:
            data_x = df[col_x].values
            data_y = df[col_y].values

    if col_z == "count":
        df = pd.DataFrame({col_x: data_x, col_y: data_y})
    else:
        # If col_z has individual measurements for each component...
        if f"{col_z} (component 0)" in df.columns:
            # If x and y also have individual measurements for each component, then use all 3 for x, y and z.
            if f"{col_x} (component 0)" in df.columns and f"{col_y} (component 0)" in df.columns:
                data_z = np.concatenate((df[f"{col_z} (component 0)"].values, df[f"{col_z} (component 1)"].values, df[f"{col_z} (component 2)"].values)) if ncomponents == "recom" else df[f"{col_z} (component 0)"].values
            # Otherwise, just use the "total" measurement.
            else: 
                data_z = df[f"{col_z} (total)"] if f"{col_z} (total)" in df.columns else df[f"{col_z}"]
        # Otherwise, just use the column as-is.
        else:
            # If x and y are measured for each component, but z isn't, then repeat the z data for each component
            if f"{col_x} (component 0)" in df.columns and f"{col_y} (component 0)" in df.columns:
                data_z = np.concatenate((df[col_z].values, df[col_z].values, df[col_z].values)) if ncomponents == "recom" else df[col_z].values
            else:
                data_z = df[col_z]
        df = pd.DataFrame({col_x: data_x, col_y: data_y, col_z: data_z})

    # Plot
    if cmap is None:
        cmap = cmap_fn(col_z)
    if vmin is None:
        vmin = vmin_fn(col_z)
    if vmax is None:
        vmax = vmax_fn(col_z)
    if xmin is None:
        xmin = vmin_fn(col_x)
    if xmax is None:
        xmax = vmax_fn(col_x)
    if ymin is None:
        ymin = vmin_fn(col_y)
    if ymax is None:
        ymax = vmax_fn(col_y)

    # If we're plotting the BPT categories, also want to show the "uncategorised" ones.
    if col_z.startswith("BPT (numeric)"):
        df_classified = df[df[col_z] > -1]
        df_unclassified = df[df[col_z] == -1]
        cmap.set_bad("white", alpha=0.0)
        histhelper(df=df_unclassified, col_x=col_x, col_y=col_y, col_z=col_z,
                    log_z=log_z, nbins=nbins, ax=ax, cmap=cmap,
                    xmin=xmin, xmax=xmax,
                    ymin=ymin, ymax=ymax,
                    vmin=vmin, vmax=vmax,
                    alpha=alpha)
        m = histhelper(df=df_classified, col_x=col_x, col_y=col_y, col_z=col_z,
                    log_z=log_z, nbins=nbins, ax=ax, cmap=cmap,
                    xmin=xmin, xmax=xmax,
                    ymin=ymin, ymax=ymax,
                    vmin=vmin, vmax=vmax,
                    alpha=alpha)
    else:
        m = histhelper(df=df, col_x=col_x, col_y=col_y, col_z=col_z, 
            log_z=log_z, nbins=nbins, ax=ax, cmap=cmap,
            xmin=xmin, xmax=xmax,
            ymin=ymin, ymax=ymax,
            vmin=vmin, vmax=vmax, 
            alpha=alpha)

    return m


###############################################################################
def plot2dcontours(df, col_x, col_y, ax,
                   nbins=100, alpha=1.0, levels=None,
                   xmin=None, xmax=None,
                   ymin=None, ymax=None,
                   linewidths=0.5, colors="k"):

    """
    For columns in which there are multiple kinematic components, include 
    ALL of them (e.g., HALPHA EW).
    """
    # Figure out how many components were fitted.
    ncomponents = "recom" if any([c.endswith("(component 2)") for c in df.columns]) else "1"

    # If either column are present as multiple components, then make a new 
    # data frame containing all of them.
    if f"{col_x} (component 0)" in df.columns:
        if f"{col_y} (component 0)" in df.columns:
            data_x = np.concatenate((df[f"{col_x} (component 0)"].values, df[f"{col_x} (component 1)"].values, df[f"{col_x} (component 2)"].values)) if ncomponents == "recom" else df[f"{col_x} (component 0)"].values
            data_y = np.concatenate((df[f"{col_y} (component 0)"].values, df[f"{col_y} (component 1)"].values, df[f"{col_y} (component 2)"].values)) if ncomponents == "recom" else df[f"{col_y} (component 0)"].values
        else:
            data_x = df[f"{col_x} (total)"] if f"{col_x} (total)" in df.columns else df[f"{col_x}"]
            data_y = df[col_y].values
    else:
        if f"{col_y} (component 0)" in df.columns:
            data_y = df[f"{col_y} (total)"] if f"{col_y} (total)" in df.columns else df[f"{col_y}"]
            data_x = df[col_x].values
        else:
            data_x = df[col_x].values
            data_y = df[col_y].values
    df = pd.DataFrame({col_x: data_x, col_y: data_y})

    # Plot
    if xmin is None:
        xmin = vmin_fn(col_x)
    if xmax is None:
        xmax = vmax_fn(col_x)
    if ymin is None:
        ymin = vmin_fn(col_y)
    if ymax is None:
        ymax = vmax_fn(col_y)
    if levels is None:
        levels = np.logspace(1, 3, 10)

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
    cuts = pd.DataFrame({"xbin": xcut, "ybin": ycut})    # Combine the x- and y-cuts
    cuts = pd.DataFrame({"xbin": xcut, "ybin": ycut})

    # Calculate the desired quantities for the data binned by x and y    
    gb_binned = df.join(cuts).groupby(list(cuts))
    df_binned = gb_binned.agg({df.columns[0]: lambda g: g.count()})

    # Pull out arrays to plot
    count_map = df_binned[df.columns[0]].values.reshape((nbins, nbins))

    # Plot.
    m = ax.contour(xbins[:-1] + dx / 2, ybins[:-1] + dy / 2, count_map.T,
                   levels=levels, colors=colors, alpha=alpha, 
                   linewidths=linewidths)

    return m


###############################################################################
def plot2dhistcontours(df, col_x, col_y, col_z=None, log_z=False,
                       vmin=None, vmax=None,
                       xmin=None, xmax=None, ymin=None, ymax=None,
                       nbins=100, ax=None, axis_labels=True, plot_colorbar=True,
                       cax=None, cax_orientation="vertical", alpha=1.0,
                       hist=True, contours=True, levels=None, linewidths=0.5, 
                       colors="k", cmap=None,
                       figsize=(9, 7)):
    """
    Plot a 2D histogram of the SAMI galaxies.
    Optionally, over-plot spaxels from a SAMI galaxy (as specified by gal) or 
    from a given Pandas DataFrame df_gal.
    """
    if col_z is None:
        assert hist is False, "in plot_full_sample: if hist is True then col_z must be specified!"
    assert cax_orientation == "horizontal" or cax_orientation == "vertical", "cax_orientation must be either 'horizontal' or 'vertical'!"

    # If no axis is specified then create a new one with a vertical colorbar.
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    else:
        fig = ax.get_figure()

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

    # Plot the full sample
    if hist:
        m = plot2dhist(df=df, col_x=col_x, col_y=col_y, col_z=col_z, log_z=log_z, nbins=nbins, ax=ax, alpha=alpha, cmap=cmap,
                       vmin=vmin, vmax=vmax, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    if contours:
        plot2dcontours(df=df, col_x=col_x, col_y=col_y, ax=ax, alpha=alpha, nbins=nbins, linewidths=linewidths, colors=colors,
                       levels=levels, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    # Decorations        
    if plot_colorbar and hist:
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

    if axis_labels:
        ax.set_xlabel(label_fn(col_x))
        ax.set_ylabel(label_fn(col_y))

    # Add the EW classification lines of Lacerda+2017.
    if col_y.startswith("log HALPHA EW"):
        # Classification lines of Lacerda+2017
        ax.axhline(np.log10(3), linestyle="--", linewidth=1, color="k")
        ax.axhline(np.log10(14), linestyle="--", linewidth=1, color="k")
        # Classification lines of Cid Fernandes+2011
        ax.axhline(np.log10(0.5), linestyle=":", linewidth=1, color="k")  # "Passive" galaxies
        if col_x.startswith("log N2"):
            ax.plot([-0.4, vmax_fn(col_x)], [np.log10(6), np.log10(6)], linestyle="-", linewidth=1, color="k") # Optimal K06 dividing line between LINERs and Seyferts
            ax.axvline(-0.4, linestyle="-", linewidth=1, color="k") # Optimal K06 dividing line between SF and other mechanisms
        else:
            ax.axhline(np.log10(6), linestyle="-", linewidth=1, color="k")  # Seyfert vs. LINER

    elif col_x.startswith("log HALPHA EW"):
        # Classification lines of Lacerda+2017
        ax.axvline(np.log10(3), linestyle="--", linewidth=1, color="k")
        ax.axvline(np.log10(14), linestyle="--", linewidth=1, color="k")
        # Classification lines of Cid Fernandes+2011
        ax.axvline(np.log10(0.5), linestyle=":", linewidth=1, color="k")  # "Passive" galaxies
        if col_y.startswith("log N2"):
            ax.plot([np.log10(6), np.log10(6)], [-0.4, vmax_fn(col_y)], linestyle="-", linewidth=1, color="k") # Optimal K06 dividing line between LINERs and Seyferts
            ax.axhline(-0.4, linestyle="-", linewidth=1, color="k") # Optimal K06 dividing line between SF and other mechanisms
        else:
            ax.axvline(np.log10(6), linestyle="-", linewidth=1, color="k")  # Seyfert vs. LINER

    elif col_x.startswith("sigma_gas - sigma_*"):
        # Vertical line at 0
        ax.axvline(0, linestyle="-", linewidth=1, color="k")
    elif col_y.startswith("sigma_gas - sigma_*"):
        # Vertical line at 0
        ax.axhline(0, linestyle="-", linewidth=1, color="k")
        
    return fig

###############################################################################
def plot2dscatter(df, col_x, col_y, col_z,
                  vmin=None, vmax=None, xmin=None, xmax=None, ymin=None, ymax=None,
                  ax=None, axis_labels=True, 
                  plot_colorbar=True, cax=None, cax_orientation="vertical", alpha=1.0, zorder=2,
                  errorbars=True, edgecolors="k", markerfacecolour="k", marker="o", markersize=20, figsize=(9, 7)):
    """
    Plot a 2D histogram of spaxels from a SAMI galaxy (as specified by gal) 
    or from a given Pandas DataFrame df_gal.
    """
    assert col_z != "count", f"{col_z} cannot be 'count' in a scatter plot!"
    if col_z is not None:
        for col in [col_x, col_y, col_z]:
            assert (col in df.columns) or (f"{col} (component 0)" in df.columns) or (f"{col} (total)" in df.columns), f"{col} is not a valid column!"            
    assert cax_orientation == "horizontal" or cax_orientation == "vertical", "cax_orientation must be either 'horizontal' or 'vertical'!"
    if col_z is None and plot_colorbar == True:
        print("WARNING: not plotting colourbar because col_z is not specified!")

    # If no axis is specified then create a new one with a vertical colorbar.
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    else:
        fig = ax.get_figure()

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
        if col_x_err_label in df:
            xerr = df[col_x_err_label]
            ax.errorbar(x=df[col_x], y=df[col_y], xerr=df[col_x_err_label],
                        ls="none", mec="none", ecolor="k", elinewidth=0.5, alpha=alpha / 2, zorder=zorder)
        # Asymmetric errors
        elif col_x_err_lower_label in df and col_x_err_upper_label in df:
            # Need to deal with special case where there is only one data point to prevent weird numpy error...
            if len(df[col_x_err_lower_label]) > 1:
                xerr = np.array([df[col_x_err_lower_label], df[col_x_err_lower_label]])
            else:
                xerr = np.array([df[col_x_err_lower_label].values[0], df[col_x_err_lower_label].values[0]])[:, None]
            ax.errorbar(x=df[col_x], y=df[col_y], xerr=xerr,
                        ls="none", mec="none", ecolor="k", elinewidth=0.5, alpha=alpha / 2, zorder=zorder)

        # y-axis: add errorbars, if they exist
        if col_y_err_label in df:
            ax.errorbar(x=df[col_x], y=df[col_y], yerr=df[col_y_err_label],
                        ls="none", mec="none", ecolor="k", elinewidth=0.5, alpha=alpha / 2, zorder=zorder)
        elif col_y_err_lower_label in df and col_y_err_upper_label in df:
            # Need to deal with special case where there is only one data point to prevent weird numpy error...
            if len(df[col_y_err_lower_label]) > 1:
                yerr = np.array([df[col_y_err_lower_label], df[col_y_err_upper_label]])
            else:
                yerr = np.array([df[col_y_err_lower_label].values[0], df[col_y_err_upper_label].values[0]])[:, None]
            ax.errorbar(x=df[col_x], y=df[col_y], yerr=yerr,
                        ls="none", mec="none", ecolor="k", elinewidth=0.5, alpha=alpha / 2, zorder=zorder)

    # Only colour the spaxels of the galaxy. 
    if col_z is not None:
        if vmin is None:
            vmin = vmin_fn(col_z)
        if vmax is None:
            vmax = vmax_fn(col_z) 
    if xmin is None:
        xmin = vmin_fn(col_x)
    if xmax is None:
        xmax = vmax_fn(col_x)
    if ymin is None:
        ymin = vmin_fn(col_y)
    if ymax is None:
        ymax = vmax_fn(col_y)  
    
    # Plot the scatter points
    if col_z is not None:
        m = ax.scatter(x=df[col_x], y=df[col_y], c=df[col_z], cmap=cmap_fn(col_z), vmin=vmin, vmax=vmax, marker=marker, edgecolors=edgecolors, s=markersize, alpha=alpha, zorder=zorder + 1)
    else:
        m = ax.scatter(x=df[col_x], y=df[col_y], c=markerfacecolour, marker=marker, edgecolors=edgecolors, s=markersize, alpha=alpha, zorder=zorder + 1)

    # Nice axis limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    if plot_colorbar and col_z is not None:
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

    if axis_labels:
        ax.set_xlabel(label_fn(col_x))
        ax.set_ylabel(label_fn(col_y))

    # Add the EW classification lines of Lacerda+2017.
    if col_y.startswith("log HALPHA EW"):
        # Classification lines of Lacerda+2017
        ax.axhline(np.log10(3), linestyle="--", linewidth=1, color="k")
        ax.axhline(np.log10(14), linestyle="--", linewidth=1, color="k")
        # Classification lines of Cid Fernandes+2011
        ax.axhline(np.log10(0.5), linestyle=":", linewidth=1, color="k")  # "Passive" galaxies
        if col_x.startswith("log N2"):
            ax.plot([-0.4, vmax_fn(col_x)], [np.log10(6), np.log10(6)], linestyle="-", linewidth=1, color="k") # Optimal K06 dividing line between LINERs and Seyferts
            ax.axvline(-0.4, linestyle="-", linewidth=1, color="k") # Optimal K06 dividing line between SF and other mechanisms
        else:
            ax.axhline(np.log10(6), linestyle="-", linewidth=1, color="k")  # "Passive" galaxies
    
    elif col_x.startswith("log HALPHA EW"):
        # Classification lines of Lacerda+2017
        ax.axvline(np.log10(3), linestyle="--", linewidth=1, color="k")
        ax.axvline(np.log10(14), linestyle="--", linewidth=1, color="k")
        # Classification lines of Cid Fernandes+2011
        ax.axvline(np.log10(0.5), linestyle=":", linewidth=1, color="k")  # "Passive" galaxies
        if col_y.startswith("log N2"):
            ax.plot([np.log10(6), np.log10(6)], [-0.4, vmax_fn(col_y)], linestyle="-", linewidth=1, color="k") # Optimal K06 dividing line between LINERs and Seyferts
            ax.axhline(-0.4, linestyle="-", linewidth=1, color="k") # Optimal K06 dividing line between SF and other mechanisms
        else:
            ax.axvline(np.log10(6), linestyle="-", linewidth=1, color="k")  # "Passive" galaxies

    elif col_x.startswith("sigma_gas - sigma_*"):
        # Vertical line at 0
        ax.axvline(0, linestyle="-", linewidth=1, color="k")
    elif col_y.startswith("sigma_gas - sigma_*"):
        # Vertical line at 0
        ax.axhline(0, linestyle="-", linewidth=1, color="k")

    return fig
