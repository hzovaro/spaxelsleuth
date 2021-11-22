import pandas as pd 
import os
import numpy as np
import copy

from spaxelsleuth.loaddata.linefns import Kewley2001, Kewley2006, Kauffman2003, Law2021_1sigma, Law2021_3sigma, ratio_fn
from spaxelsleuth.plotting.plottools import plot_empty_BPT_diagram

from IPython.core.debugger import Tracer


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc, rcParams
from matplotlib.lines import Line2D
rc("text", usetex=False)
rc("font",**{"family": "serif", "size": 12})
plt.ion()

##############################################################################
GRID_PATH = "../data/grids/"

##############################################################################
# Convert solar-scaled abundances to log(O/H) + 12
zeta_to_logOH12 = {
    "0.05": 7.6954,  # GC_ZO_0086.abn
    "0.20": 8.2808,  # GC_ZO_0332.abn
    "0.40": 8.5422,  # GC_ZO_0606.abn
    "1.00": 8.8879,  # GC_ZO_1342.abn
    "2.00": 9.1636,  # GC_ZO_2533.abn
}

###############################################################################
def load_HII_grid(drrecmode=0):
    # Load the dataframe
    grid_df = pd.read_hdf(os.path.join(GRID_PATH, "HII_grid.hd5"))
    grid_df.drrecmode = grid_df.drrecmode.astype(str)

    assert drrecmode in grid_df["drrecmode"].unique(), f"drrecmode {drrecmode} not in pAGB dataframe!"

    # Sort grid
    grid_df = grid_df.sort_values("log(U) (inner)")
    grid_df = grid_df.sort_values("log(P/k)")
    grid_df = grid_df.sort_values("Nebular abundance (Z/Zsun)")

    # Convert solar-scaled abundances to log(O/H) + 12
    grid_df["log(O/H) + 12"] = [zeta_to_logOH12[f"{zeta:.2f}"] for zeta in grid_df["Nebular abundance (Z/Zsun)"]]

    # Evaluate line ratios
    grid_df = ratio_fn(grid_df)

    # Separate into 2 grids
    return grid_df[grid_df.drrecmode == str(drrecmode)]


###############################################################################
def load_pAGB_grid(drrecmode="0"):
    # Load the dataframe
    grid_df = pd.read_hdf(os.path.join(GRID_PATH, "pAGB_grid.hd5"))
    grid_df.drrecmode = grid_df.drrecmode.astype(str)

    assert drrecmode in grid_df["drrecmode"].unique(), f"drrecmode {drrecmode} not in HII region dataframe!"

    # Sort grid
    grid_df = grid_df.sort_values("log(U) (inner)")
    grid_df = grid_df.sort_values("log(P/k)")
    grid_df = grid_df.sort_values("Nebular abundance (Z/Zsun)")

    # Grid variables
    logU_vals = grid_df.loc[:, "log(U) (inner)"].unique()
    zeta_vals = grid_df.loc[:, "Nebular abundance (Z/Zsun)"].unique()
    logPk_vals = grid_df.loc[:, "log(P/k)"].unique()

    # Sort parameter arrays
    logU_vals.sort()
    zeta_vals.sort()
    logPk_vals.sort()

    # Convert solar-scaled abundances to log(O/H) + 12
    grid_df["log(O/H) + 12"] = [zeta_to_logOH12[f"{zeta:.2f}"] for zeta in grid_df["Nebular abundance (Z/Zsun)"]]

    # Evaluate line ratios
    grid_df = ratio_fn(grid_df)

    # Separate into 2 grids
    return grid_df[grid_df.drrecmode == str(drrecmode)]


###############################################################################
def load_AGN_grid(drrecmode="0"):
    # Load the dataframe
    grid_df = pd.read_hdf(os.path.join(GRID_PATH, "AGN_grid.hd5"))
    grid_df.drrecmode = grid_df.drrecmode.astype(str)
    grid_df["Nebular abundance (Z/Zsun)"] = grid_df["Nebular abundance (Z/Zsun)"].astype(float)
    assert drrecmode in grid_df["drrecmode"].unique(), f"drrecmode {drrecmode} not in AGN dataframe!"

    # Sort grid
    grid_df = grid_df.sort_values("log(U) (inner)")
    grid_df = grid_df.sort_values("log(P/k)")
    grid_df = grid_df.sort_values("Nebular abundance (Z/Zsun)")
    grid_df = grid_df.sort_values("E_peak (log_10(keV))")

    # Convert solar-scaled abundances to log(O/H) + 12
    grid_df["log(O/H) + 12"] = [zeta_to_logOH12[f"{zeta:.2f}"] for zeta in grid_df["Nebular abundance (Z/Zsun)"]]

    # Evaluate line ratios
    grid_df = ratio_fn(grid_df)

    # Separate into 2 grids
    return grid_df[grid_df.drrecmode == str(drrecmode)]

###############################################################################
def load_shock_grid(model_type):
    assert model_type in ["total", "precursor", "shock"], "type must be one of total, precursor or shock!"

    # Load the dataframe
    grid_df = pd.read_csv(os.path.join(GRID_PATH, "MAPPINGS_shock_grid_13_fluxes.csv"))
    grid_df = grid_df.rename(columns={"density": "Density (cm^-3)",
                                      "v_s": "v_shock (km s^-1)",
                                      "lgAlpha_0": "log_10(P_mag/P_gas)",
                                      "abund": "Nebular abundance (percentage solar)",
                                      "Gridpoint": "Model"})
    grid_df["Nebular abundance (Z/Zsun)"] = grid_df["Nebular abundance (percentage solar)"] / 100.0

    # Keep only one flux type
    if model_type == "total":
        type_str = "tot"
    elif model_type == "precursor":
        type_str = "pre"
    elif model_type == "shock":
        type_str = "shc"

    if model_type != "total":
        grid_df = grid_df.drop(columns=[col for col in grid_df.columns if col.endswith("_tot")])
    if model_type != "precursor":
        grid_df = grid_df.drop(columns=[col for col in grid_df.columns if col.endswith("_pre")])
    if model_type != "shock":
        grid_df = grid_df.drop(columns=[col for col in grid_df.columns if col.endswith("_shc")])

    rename_dict = {}
    for col in [c for c in grid_df.columns if c.endswith(type_str)]:
        rename_dict[col] = col.split(f"_{type_str}")[0]
    grid_df = grid_df.rename(columns=rename_dict)
    grid_df = grid_df.rename(columns={"Halpha": "HALPHA", "Hbeta": "HBETA"})

    # Sort grid
    grid_df = grid_df.sort_values("v_shock (km s^-1)")
    grid_df = grid_df.sort_values("Density (cm^-3)")
    grid_df = grid_df.sort_values("Nebular abundance (Z/Zsun)")
    grid_df = grid_df.sort_values("log_10(P_mag/P_gas)")

    # Convert solar-scaled abundances to log(O/H) + 12
    grid_df["log(O/H) + 12"] = 8.76 + np.log10(grid_df["Nebular abundance (Z/Zsun)"])

    # Evaluate line ratios
    grid_df = ratio_fn(grid_df)

    return grid_df

###############################################################################
def plot_BPT_with_grids(grid="HII",
                        logPk=5.0,
                        logOH12_AGN=8.8879, logOH12_shock=8.76,
                        drrecmode="0", shock_model_type="total",
                        cmap_logU="cubehelix", cmap_logOH12="cividis", cmap_logEpeak="spring",
                        cmap_vel="plasma", cmap_density="summer_r",
                        ls="-", lw=1, zorder=1, colorbar=False):

    # Empty BPT diagram
    fig, axs = plot_empty_BPT_diagram(colorbar=colorbar)
    
    # for legend
    lines = []
    labels = []

    ###########################################################################
    if grid == "HII":
        # Load the grid
        grid_df = load_HII_grid(drrecmode=drrecmode)

        # Grid variables
        logU_vals = grid_df.loc[:, "log(U) (inner)"].unique()
        logOH12_vals = grid_df.loc[:, "log(O/H) + 12"].unique().astype(float)
        logPk_vals = grid_df.loc[:, "log(P/k)"].unique()
        logU_vals.sort()
        logOH12_vals.sort()
        logPk_vals.sort()

        # Colormaps
        cmap_logU = plt.cm.get_cmap(cmap_logU, len(logU_vals))
        cmap_logOH12 = plt.cm.get_cmap(cmap_logOH12, len(logOH12_vals))
        logU_lw = lw
        logOH12_lw = lw

        # Manually add legend entries
        lines += [Line2D([0], [0], color=cmap_logOH12(ii), linewidth=logOH12_lw, linestyle=ls) for ii in range(len(logOH12_vals))]
        labels += [r"$\log({\rm O/H}) + 12 = %.2f$" % logOH12 for logOH12 in logOH12_vals]
        lines += [Line2D([0], [0], color=cmap_logU(ii), linewidth=logU_lw, linestyle=ls) for ii in range(len(logU_vals))]
        labels += [r"$\log(U) = %.1f$" % logU for logU in logU_vals]

        # label to indicate pressure
        axs[0].text(s=r"HII region models, $\log_{10}(P/k) = %.2f$" % logPk, x=0.1, y=1.05, transform=axs[0].transAxes)

        # Plot
        for ii, xratio in enumerate(["log N2", "log S2", "log O1"]):
            # Plot model grids
            for logU_idx, logU in enumerate(logU_vals):
                # Lines of constant log(Z)
                cond = grid_df["log(P/k)"] == logPk
                cond &= grid_df["log(U) (inner)"] == logU
                df1 = grid_df[cond].sort_values("log(O/H) + 12")
                axs[ii].plot(df1.loc[:, xratio].values, df1.loc[:, "log O3"].values, ls=ls, c=cmap_logU(logU_idx), zorder=zorder, linewidth=logU_lw)

            for logOH12_idx, logOH12 in enumerate(logOH12_vals):
                # Lines of constant log(U)
                cond = grid_df["log(O/H) + 12"] == logOH12
                cond &= grid_df["log(P/k)"] == logPk
                df1 = grid_df[cond].sort_values("log(U) (inner)")
                axs[ii].plot(df1.loc[:, xratio].values, df1.loc[:, "log O3"].values, ls=ls, c=cmap_logOH12(logOH12_idx), zorder=zorder, linewidth=logOH12_lw)
                
        # Shrink current axis by 40% to add legend
        for ax in axs:
            box_ax = ax.get_position()
            ax.set_position([box_ax.x0, box_ax.y0 + 0.25 * box_ax.height, box_ax.width, box_ax.height * 0.66])
        axs[1].legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=6, prop={"size": "small"})

        return fig, axs

    ###########################################################################
    if grid == "pAGB":
        # Load the grid
        grid_df = load_pAGB_grid(drrecmode=drrecmode)
        assert len(grid_df["age (yr)"].unique()) == 1, "pAGB grid contains multiple ages!"
        assert len(grid_df["Stellar abundance (log(Z/Zsun))"].unique()) == 1, "pAGB grid contains multiple stellar abundances!"

        age_gyr = grid_df["age (yr)"].unique()[0] / 1e9
        logZ_star = grid_df["Stellar abundance (log(Z/Zsun))"].unique()[0]

        # Grid variables
        logU_vals = grid_df.loc[:, "log(U) (inner)"].unique()
        logOH12_vals = grid_df.loc[:, "log(O/H) + 12"].unique().astype(float)
        logPk_vals = grid_df.loc[:, "log(P/k)"].unique()
        logU_vals.sort()
        logOH12_vals.sort()
        logPk_vals.sort()

        # Colormaps
        cmap_logU = plt.cm.get_cmap(cmap_logU, len(logU_vals))
        cmap_logOH12 = plt.cm.get_cmap(cmap_logOH12, len(logOH12_vals))
        logU_lw = lw
        logOH12_lw = lw

        # Manually add legend entries
        lines += [Line2D([0], [0], color=cmap_logOH12(ii), linewidth=logOH12_lw, linestyle=ls) for ii in range(len(logOH12_vals))]
        labels += [r"$\log({\rm O/H}) + 12 = %.2f$" % logOH12 for logOH12 in logOH12_vals]
        lines += [Line2D([0], [0], color=cmap_logU(ii), linewidth=logU_lw, linestyle=ls) for ii in range(len(logU_vals))]
        labels += [r"$\log(U) = %.1f$" % logU for logU in logU_vals]

        # label to fixed model parameters
        axs[0].text(s=r"pAGB models, $\log_{10}(P/k) = %.2f,\, t = %.2f\,{\rm Gyr},\,\log[Z/Z_\odot] = %.2f$" % (logPk, age_gyr, logZ_star), x=0.1, y=1.05, transform=axs[0].transAxes)

        # Plot
        for ii, xratio in enumerate(["log N2", "log S2", "log O1"]):
            # Plot model grids
            for logU_idx, logU in enumerate(logU_vals):
                # Lines of constant log(Z)
                cond = grid_df["log(P/k)"] == logPk
                cond &= grid_df["log(U) (inner)"] == logU
                df1 = grid_df[cond].sort_values("log(O/H) + 12")
                axs[ii].plot(df1.loc[:, xratio].values, df1.loc[:, "log O3"].values, ls=ls, c=cmap_logU(logU_idx), zorder=zorder, linewidth=logU_lw)

            for logOH12_idx, logOH12 in enumerate(logOH12_vals):
                # Lines of constant log(U)
                cond = grid_df["log(O/H) + 12"] == logOH12
                cond &= grid_df["log(P/k)"] == logPk
                df1 = grid_df[cond].sort_values("log(U) (inner)")
                axs[ii].plot(df1.loc[:, xratio].values, df1.loc[:, "log O3"].values, ls=ls, c=cmap_logOH12(logOH12_idx), zorder=zorder, linewidth=logOH12_lw)
                
        # Shrink current axis by 40% to add legend
        for ax in axs:
            box_ax = ax.get_position()
            ax.set_position([box_ax.x0, box_ax.y0 + 0.25 * box_ax.height, box_ax.width, box_ax.height * 0.66])
        axs[1].legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=6, prop={"size": "small"})

        return fig, axs

    ###########################################################################
    if grid == "AGN":
        # Load the grid
        grid_df = load_AGN_grid(drrecmode=drrecmode)
        assert len(grid_df["Gamma"].unique()) == 1, "AGN grid contains Gamma values!"
        assert len(grid_df["p_NT"].unique()) == 1, "AGN grid contains p_NT values!"
        Gamma = grid_df["Gamma"].unique()[0]
        p_NT = grid_df["p_NT"].unique()[0]

        # Grid variables
        logU_vals = grid_df.loc[:, "log(U) (inner)"].unique()
        logOH12_vals = grid_df.loc[:, "log(O/H) + 12"].unique().astype(float)
        logPk_vals = grid_df.loc[:, "log(P/k)"].unique()
        logEpeak_vals = grid_df.loc[:, "E_peak (log_10(keV))"].unique()
        logU_vals.sort()
        logOH12_vals.sort()
        logPk_vals.sort()
        logEpeak_vals.sort()

        # Colormaps
        cmap_logU = plt.cm.get_cmap(cmap_logU, len(logU_vals))
        cmap_logEpeak = plt.cm.get_cmap(cmap_logEpeak, len(logEpeak_vals))
        logU_lw = lw
        logEpeak_lw = lw

        # Manually add legend entries
        lines += [Line2D([0], [0], color=cmap_logEpeak(ii), linewidth=logEpeak_lw, linestyle=ls) for ii in range(len(logEpeak_vals))]
        labels += [r"$\log(E_{\rm peak} \, [{\rm keV}]) = %.2f$" % logEpeak for logEpeak in logEpeak_vals]
        lines += [Line2D([0], [0], color=cmap_logU(ii), linewidth=logU_lw, linestyle=ls) for ii in range(len(logU_vals))]
        labels += [r"$\log(U) = %.1f$" % logU for logU in logU_vals]

        # label to fixed model parameters
        axs[0].text(s=r"AGN models, $\log({\rm O/H}) + 12 = %.2f,\,\log_{10}(P/k) = %.2f,\, \Gamma = %.2f,\,p_{\rm NT} = %.2f$" % (logOH12_AGN, logPk, Gamma, p_NT), x=0.1, y=1.05, transform=axs[0].transAxes)

        # Plot
        for ii, xratio in enumerate(["log N2", "log S2", "log O1"]):
            # Plot model grids
            for logU_idx, logU in enumerate(logU_vals):
                # Lines of constant log(Z)
                cond = grid_df["log(O/H) + 12"] == logOH12_AGN
                cond &= grid_df["log(P/k)"] == logPk
                cond &= grid_df["log(U) (inner)"] == logU
                df1 = grid_df[cond].sort_values("E_peak (log_10(keV))")
                axs[ii].plot(df1.loc[:, xratio].values, df1.loc[:, "log O3"].values, ls=ls, c=cmap_logU(logU_idx), zorder=zorder, linewidth=logU_lw)

            for logEpeak_idx, logEpeak in enumerate(logEpeak_vals):
                # Lines of constant log(E_peak)
                cond = grid_df["log(O/H) + 12"] == logOH12_AGN
                cond &= grid_df["log(P/k)"] == logPk
                cond &= grid_df["E_peak (log_10(keV))"] == logEpeak
                df1 = grid_df[cond].sort_values("log(U) (inner)")
                axs[ii].plot(df1.loc[:, xratio].values, df1.loc[:, "log O3"].values, ls=ls, c=cmap_logEpeak(logEpeak_idx), zorder=zorder, linewidth=logEpeak_lw)
                
        # Shrink current axis by 40% to add legend
        for ax in axs:
            box_ax = ax.get_position()
            ax.set_position([box_ax.x0, box_ax.y0 + 0.25 * box_ax.height, box_ax.width, box_ax.height * 0.66])
        axs[1].legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=6, prop={"size": "small"})

        return fig, axs

    ###########################################################################
    if grid == "shock":
        # Load the grid
        grid_df = load_shock_grid(model_type=shock_model_type)
        assert len(grid_df["log_10(P_mag/P_gas)"].unique()) == 1, "Shock grid contains multiple magnetic parameters!"
        logAlpha0 = grid_df["log_10(P_mag/P_gas)"].unique()[0]

        # Grid variables
        vel_vals = grid_df.loc[:, "v_shock (km s^-1)"].unique()
        density_vals = grid_df.loc[:, "Density (cm^-3)"].unique().astype(float)
        logOH12_vals = grid_df.loc[:, "log(O/H) + 12"].unique()
        vel_vals.sort()
        density_vals.sort()
        logOH12_vals.sort()

        # Colormaps
        cmap_vel = plt.cm.get_cmap(cmap_vel, len(vel_vals))
        cmap_density = plt.cm.get_cmap(cmap_density, len(density_vals))
        vel_lw = lw
        density_lw = lw

        # Manually add legend entries
        lines += [Line2D([0], [0], color=cmap_density(ii), linewidth=density_lw, linestyle=ls) for ii in range(len(density_vals))]
        labels += [r"$n = %.0f\,\rm cm^{-3}$" % density for density in density_vals]
        lines += [Line2D([0], [0], color=cmap_vel(ii), linewidth=vel_lw, linestyle=ls) for ii in range(len(vel_vals))]
        labels += [r"$v_{\rm shock} = %.0f\,\rm km\,s^{-1}$" % vel for vel in vel_vals]

        # label to fixed model parameters
        axs[0].text(s=r"Shock models (%s), $\log_{10}(P/k) = %.2f,\, \log_{10}(P_{\rm mag}/P_{\rm gas}) = %.2f$" % (shock_model_type, logOH12_shock, logAlpha0), x=0.1, y=1.05, transform=axs[0].transAxes)

        # Plot
        for ii, xratio in enumerate(["log N2", "log S2", "log O1"]):
            # Plot model grids
            for vel_idx, vel in enumerate(vel_vals):
                # Lines of constant shock velocity
                cond = grid_df["log(O/H) + 12"] == logOH12_shock
                cond &= grid_df["v_shock (km s^-1)"] == vel
                df1 = grid_df[cond].sort_values("log(O/H) + 12")
                axs[ii].plot(df1.loc[:, xratio].values, df1.loc[:, "log O3"].values, ls=ls, c=cmap_vel(vel_idx), zorder=zorder, linewidth=vel_lw)

            for density_idx, density in enumerate(density_vals):
                # Lines of constant density
                cond = grid_df["Density (cm^-3)"] == density
                cond &= grid_df["log(O/H) + 12"] == logOH12_shock
                df1 = grid_df[cond].sort_values("v_shock (km s^-1)")
                axs[ii].plot(df1.loc[:, xratio].values, df1.loc[:, "log O3"].values, ls=ls, c=cmap_density(density_idx), zorder=zorder, linewidth=density_lw)
                
        # Shrink current axis by 40% to add legend
        for ax in axs:
            box_ax = ax.get_position()
            ax.set_position([box_ax.x0, box_ax.y0 + 0.25 * box_ax.height, box_ax.width, box_ax.height * 0.66])
        axs[1].legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=6, prop={"size": "small"})

        return fig, axs


###############################################################################
def plot_grids(col_x, col_y, ax=None,
               grid="HII",
               logPk=5.0,
               logOH12_AGN=8.8879, logOH12_shock=8.76,
               drrecmode="0", shock_model_type="total",
               cmap_logU="cubehelix", cmap_logOH12="cividis", cmap_logEpeak="spring",
               cmap_vel="plasma", cmap_density="summer_r",
               alpha=1.0, ls="-", lw=1, zorder=1, shrink_axis=True, legend=True, return_legend_handles=False):

    # for legend
    lines = []
    labels = []

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    else:
        fig = ax.get_figure()

    ###########################################################################
    if grid == "HII":
        # Load the grid
        grid_df = load_HII_grid(drrecmode=drrecmode)

        assert col_x in grid_df.columns, f"{col_x} not found in DataFrame!"
        assert col_y in grid_df.columns, f"{col_y} not found in DataFrame!"

        # Grid variables
        logU_vals = grid_df.loc[:, "log(U) (inner)"].unique()
        logOH12_vals = grid_df.loc[:, "log(O/H) + 12"].unique().astype(float)
        logPk_vals = grid_df.loc[:, "log(P/k)"].unique()
        logU_vals.sort()
        logOH12_vals.sort()
        logPk_vals.sort()

        # Colormaps
        cmap_logU = plt.cm.get_cmap(cmap_logU, len(logU_vals))
        cmap_logOH12 = plt.cm.get_cmap(cmap_logOH12, len(logOH12_vals))
        logU_lw = lw
        logOH12_lw = lw

        # Manually add legend entries
        lines += [Line2D([0], [0], color=cmap_logOH12(ii), linewidth=logOH12_lw, linestyle=ls) for ii in range(len(logOH12_vals))]
        labels += [r"$\log({\rm O/H}) + 12 = %.2f$" % logOH12 for logOH12 in logOH12_vals]
        lines += [Line2D([0], [0], color=cmap_logU(ii), linewidth=logU_lw, linestyle=ls) for ii in range(len(logU_vals))]
        labels += [r"$\log(U) = %.1f$" % logU for logU in logU_vals]

        # Plot model grids
        for logU_idx, logU in enumerate(logU_vals):
            # Lines of constant log(Z)
            cond = grid_df["log(P/k)"] == logPk
            cond &= grid_df["log(U) (inner)"] == logU
            df1 = grid_df[cond].sort_values("log(O/H) + 12")
            ax.plot(df1.loc[:, col_x].values, df1.loc[:, col_y].values, ls=ls, c=cmap_logU(logU_idx), zorder=zorder, linewidth=logU_lw, alpha=alpha)

        for logOH12_idx, logOH12 in enumerate(logOH12_vals):
            # Lines of constant log(U)
            cond = grid_df["log(O/H) + 12"] == logOH12
            cond &= grid_df["log(P/k)"] == logPk
            df1 = grid_df[cond].sort_values("log(U) (inner)")
            ax.plot(df1.loc[:, col_x].values, df1.loc[:, col_y].values, ls=ls, c=cmap_logOH12(logOH12_idx), zorder=zorder, linewidth=logOH12_lw, alpha=alpha)
        
        title=r"HII region models" + "\n" + r"$\log_{10}(P/k) = %.2f$" % logPk

    ###########################################################################
    if grid == "pAGB":
        # Load the grid
        grid_df = load_pAGB_grid(drrecmode=drrecmode)
        assert len(grid_df["age (yr)"].unique()) == 1, "pAGB grid contains multiple ages!"
        assert len(grid_df["Stellar abundance (log(Z/Zsun))"].unique()) == 1, "pAGB grid contains multiple stellar abundances!"

        age_gyr = grid_df["age (yr)"].unique()[0] / 1e9
        logZ_star = grid_df["Stellar abundance (log(Z/Zsun))"].unique()[0]

        # Grid variables
        logU_vals = grid_df.loc[:, "log(U) (inner)"].unique()
        logOH12_vals = grid_df.loc[:, "log(O/H) + 12"].unique().astype(float)
        logPk_vals = grid_df.loc[:, "log(P/k)"].unique()
        logU_vals.sort()
        logOH12_vals.sort()
        logPk_vals.sort()

        # Colormaps
        cmap_logU = plt.cm.get_cmap(cmap_logU, len(logU_vals))
        cmap_logOH12 = plt.cm.get_cmap(cmap_logOH12, len(logOH12_vals))
        logU_lw =lw
        logOH12_lw =lw

        # Manually add legend entries
        lines += [Line2D([0], [0], color=cmap_logOH12(ii), linewidth=logOH12_lw, linestyle=ls) for ii in range(len(logOH12_vals))]
        labels += [r"$\log({\rm O/H}) + 12 = %.2f$" % logOH12 for logOH12 in logOH12_vals]
        lines += [Line2D([0], [0], color=cmap_logU(ii), linewidth=logU_lw, linestyle=ls) for ii in range(len(logU_vals))]
        labels += [r"$\log(U) = %.1f$" % logU for logU in logU_vals]

        # Plot model grids
        for logU_idx, logU in enumerate(logU_vals):
            # Lines of constant log(Z)
            cond = grid_df["log(P/k)"] == logPk
            cond &= grid_df["log(U) (inner)"] == logU
            df1 = grid_df[cond].sort_values("log(O/H) + 12")
            ax.plot(df1.loc[:, col_x].values, df1.loc[:, col_y].values, ls=ls, c=cmap_logU(logU_idx), zorder=zorder, linewidth=logU_lw, alpha=alpha)

        for logOH12_idx, logOH12 in enumerate(logOH12_vals):
            # Lines of constant log(U)
            cond = grid_df["log(O/H) + 12"] == logOH12
            cond &= grid_df["log(P/k)"] == logPk
            df1 = grid_df[cond].sort_values("log(U) (inner)")
            ax.plot(df1.loc[:, col_x].values, df1.loc[:, col_y].values, ls=ls, c=cmap_logOH12(logOH12_idx), zorder=zorder, linewidth=logOH12_lw, alpha=alpha)
                
        title=r"pAGB models" + "\n" + r"$\log_{10}(P/k) = %.2f$" % logPk + "\n" + r"$t = %.2f\,{\rm Gyr}$" % age_gyr + "\n" + r"$\log[Z/Z_\odot] = %.2f$" % logZ_star

    ###########################################################################
    if grid == "AGN":
        # Load the grid
        grid_df = load_AGN_grid(drrecmode=drrecmode)
        assert len(grid_df["Gamma"].unique()) == 1, "AGN grid contains Gamma values!"
        assert len(grid_df["p_NT"].unique()) == 1, "AGN grid contains p_NT values!"
        Gamma = grid_df["Gamma"].unique()[0]
        p_NT = grid_df["p_NT"].unique()[0]

        # Grid variables
        logU_vals = grid_df.loc[:, "log(U) (inner)"].unique()
        logOH12_vals = grid_df.loc[:, "log(O/H) + 12"].unique().astype(float)
        logPk_vals = grid_df.loc[:, "log(P/k)"].unique()
        logEpeak_vals = grid_df.loc[:, "E_peak (log_10(keV))"].unique()
        logU_vals.sort()
        logOH12_vals.sort()
        logPk_vals.sort()
        logEpeak_vals.sort()

        # Colormaps
        cmap_logU = plt.cm.get_cmap(cmap_logU, len(logU_vals))
        cmap_logEpeak = plt.cm.get_cmap(cmap_logEpeak, len(logEpeak_vals))
        logU_lw =lw
        logEpeak_lw =lw

        # Manually add legend entries
        lines += [Line2D([0], [0], color=cmap_logEpeak(ii), linewidth=logEpeak_lw, linestyle=ls) for ii in range(len(logEpeak_vals))]
        labels += [r"$\log(E_{\rm peak} \, [{\rm keV}]) = %.2f$" % logEpeak for logEpeak in logEpeak_vals]
        lines += [Line2D([0], [0], color=cmap_logU(ii), linewidth=logU_lw, linestyle=ls) for ii in range(len(logU_vals))]
        labels += [r"$\log(U) = %.1f$" % logU for logU in logU_vals]

        # Plot model grids
        for logU_idx, logU in enumerate(logU_vals):
            # Lines of constant log(Z)
            cond = grid_df["log(O/H) + 12"] == logOH12_AGN
            cond &= grid_df["log(P/k)"] == logPk
            cond &= grid_df["log(U) (inner)"] == logU
            df1 = grid_df[cond].sort_values("E_peak (log_10(keV))")
            ax.plot(df1.loc[:, col_x].values, df1.loc[:, col_y].values, ls=ls, c=cmap_logU(logU_idx), zorder=zorder, linewidth=logU_lw, alpha=alpha)

        for logEpeak_idx, logEpeak in enumerate(logEpeak_vals):
            # Lines of constant log(E_peak)
            cond = grid_df["log(O/H) + 12"] == logOH12_AGN
            cond &= grid_df["log(P/k)"] == logPk
            cond &= grid_df["E_peak (log_10(keV))"] == logEpeak
            df1 = grid_df[cond].sort_values("log(U) (inner)")
            ax.plot(df1.loc[:, col_x].values, df1.loc[:, col_y].values, ls=ls, c=cmap_logEpeak(logEpeak_idx), zorder=zorder, linewidth=logEpeak_lw, alpha=alpha)

        title=r"AGN models" + "\n" + r"$\log({\rm O/H}) + 12 = %.2f$" % logOH12_AGN + "\n" + r"$\log_{10}(P/k) = %.2f$" % logPk + "\n" + r"$\Gamma = %.2f$" % Gamma + "\n" + r"$p_{\rm NT} = %.2f$" % p_NT

    ###########################################################################
    if grid == "shock":
        # Load the grid
        grid_df = load_shock_grid(model_type=shock_model_type)
        assert col_x in grid_df.columns, f"{col_x} not found in DataFrame!"
        assert col_y in grid_df.columns, f"{col_y} not found in DataFrame!"
        assert len(grid_df["log_10(P_mag/P_gas)"].unique()) == 1, "Shock grid contains multiple magnetic parameters!"
        logAlpha0 = grid_df["log_10(P_mag/P_gas)"].unique()[0]

        # Grid variables
        vel_vals = grid_df.loc[:, "v_shock (km s^-1)"].unique()
        density_vals = grid_df.loc[:, "Density (cm^-3)"].unique().astype(float)
        logOH12_vals = grid_df.loc[:, "log(O/H) + 12"].unique()
        vel_vals.sort()
        density_vals.sort()
        logOH12_vals.sort()

        # Colormaps
        cmap_vel = plt.cm.get_cmap(cmap_vel, len(vel_vals))
        cmap_density = plt.cm.get_cmap(cmap_density, len(density_vals))
        vel_lw = lw
        density_lw = lw

        # Manually add legend entries
        lines += [Line2D([0], [0], color=cmap_density(ii), linewidth=density_lw, linestyle=ls) for ii in range(len(density_vals))]
        labels += [r"$n = %.0f\,\rm cm^{-3}$" % density for density in density_vals]
        lines += [Line2D([0], [0], color=cmap_vel(ii), linewidth=vel_lw, linestyle=ls) for ii in range(len(vel_vals))]
        labels += [r"$v_{\rm shock} = %.0f\,\rm km\,s^{-1}$" % vel for vel in vel_vals]

        # Plot model grids
        for vel_idx, vel in enumerate(vel_vals):
            # Lines of constant shock velocity
            cond = grid_df["log(O/H) + 12"] == logOH12_shock
            cond &= grid_df["v_shock (km s^-1)"] == vel
            df1 = grid_df[cond].sort_values("log(O/H) + 12")
            ax.plot(df1.loc[:, col_x].values, df1.loc[:, col_y].values, ls=ls, c=cmap_vel(vel_idx), zorder=zorder, linewidth=vel_lw, alpha=alpha)

        for density_idx, density in enumerate(density_vals):
            # Lines of constant density
            cond = grid_df["Density (cm^-3)"] == density
            cond &= grid_df["log(O/H) + 12"] == logOH12_shock
            df1 = grid_df[cond].sort_values("v_shock (km s^-1)")
            ax.plot(df1.loc[:, col_x].values, df1.loc[:, col_y].values, ls=ls, c=cmap_density(density_idx), zorder=zorder, linewidth=density_lw, alpha=alpha)
                
        title = r"Shock models (%s)" % shock_model_type + "\n" + r"$\log_{10}(P/k) = %.2f$" % logOH12_shock + "\n" + r"$\log_{10}(P_{\rm mag}/P_{\rm gas}) = %.2f$" % (logAlpha0)

    ###########################################################################
    # Decorations
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    ax.grid(b=True)

    # Shrink current axis by 40% to add legend
    if shrink_axis:
        box_ax = ax.get_position()
        ax.set_position([box_ax.x0, box_ax.y0, box_ax.width * .75, box_ax.height])
    if legend:
        ax.legend(lines, labels, loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1, prop={"size": "small"},
            title=title)

    if return_legend_handles:
        return fig, ax, lines, labels
    else:
        return fig, ax
