if __name__ == "__main__":
    import os

    from spaxelsleuth import load_user_config
    try:
        load_user_config("/Users/u5708159/Desktop/spaxelsleuth_test/.myconfig.json")
    except FileNotFoundError:
        load_user_config("/home/u5708159/.spaxelsleuthconfig.json")
    from spaxelsleuth.config import settings
    from spaxelsleuth.io.io import make_df, load_df

    nthreads = 10
    ncomponents = 1
    eline_SNR_min = 1
    eline_ANR_min = 1
    correct_extinction = False
    
    # List of galaxies with LZIFU data
    gals = [int(f.split("_")[0]) for f in os.listdir(settings["lzifu"]["input_path"]) if f.endswith("1_comp.fits") and not f.startswith(".")]

    # Create the DataFrame
    make_df(survey="lzifu",
            gals=gals,
            bin_type="default",
            ncomponents=ncomponents,
            eline_SNR_min=eline_SNR_min,
            eline_ANR_min=eline_ANR_min,
            correct_extinction=correct_extinction,
            metallicity_diagnostics=[
                "N2Ha_PP04",
            ],
            sigma_inst_kms=29.6,
            nthreads=nthreads)

    # Load the DataFrames
    df, ss_params = load_df(
         survey="lzifu",
         bin_type="default",
        ncomponents=ncomponents,
        eline_SNR_min=eline_SNR_min,
        eline_ANR_min=eline_ANR_min,
        correct_extinction=correct_extinction,
    )