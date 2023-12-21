import matplotlib.pyplot as plt
from matplotlib import rc

from spaxelsleuth.plotting.plottools import get_label

rc("text", usetex=True)
plt.ion()
plt.close("all")

labels = [
    'log M_*',
    'log SFR (R_e (MGE, FD, r), resolved) (total)',
    'log SFR (R_e (MGE, FD, r), resolved) (component 1)',
    'log SFR surface density (R_e (MGE, FD, r), resolved) (total)',
    'log SFR surface density (R_e (MGE, FD, r), resolved) (component 1)',
    'Stellar age (Gyr) (R_e (MGE))',
    'Stellar [Z/H] (R_e (MGE))',
    'log sSFR (R_e (MGE, FD, r), resolved) (total)',
    'log sSFR (R_e (MGE, FD, r), resolved) (component 1)',
    'log SFR (3kpc, resolved) (total)',
    'log SFR (3kpc, resolved) (component 1)',
    'log SFR surface density (3kpc, resolved) (total)',
    'log SFR surface density (3kpc, resolved) (component 1)',
    'Stellar age (Gyr) (3kpc round)',
    'Stellar [Z/H] (3kpc round)',
    'log sSFR (3kpc, resolved) (total)',
    'log sSFR (3kpc, resolved) (component 1)',
    'i (degrees)',
    'i (MGE, FD, r) (degrees)',
    'R_e (MGE, FD, r) (kpc)',
    'log(M/R_e)',
    'log(M/R_e^2)',
    'log(M/R_e) (MGE, FD, r)',
    'log(M/R_e^2) (MGE, FD, r)',
    'log mean HALPHA A/N (R_e (MGE), resolved)',
    'log mean HALPHA A/N (3kpc, resolved)',
    'kpc per arcsec',
    'D4000',
    'HDELTA_A (R_e (MGE))',
    'G (SFR)',
]

fig, ax = plt.subplots(figsize=(5, 20))
yy = 0.05 
dy = 1. / len(labels)
for label in labels:
    ax.text(s=f"{get_label(label)}", x=0.5, y=yy, ha="center", va="bottom", transform=ax.transAxes)
    yy += dy
