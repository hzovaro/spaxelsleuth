from spaxelsleuth.plotting.plottools import get_label

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

for label in labels:
    print(f"{label}: {get_label(label)}")