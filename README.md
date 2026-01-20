# frabjous

This is a deep learning-based framework to classify distinct FRB morphologies.

The repository also includes frabjous_sim, which contains scripts and configuration files used to simulate FRB dynamic spectra of different morphological classes. These simulations are used to generate labelled datasets for training and testing the CNN models.

Using the Simulation Scripts
1. Go to the simulation directory (frabjous_sim)
2. Edit the YAML configuration file corresponding to the desired morphology to set parameter ranges (e.g. DM, scattering, drift rate, S/N, number of bursts).
3. python <simulation_script>.py
4. The script generates simulated FRB dynamic spectra along with metadata, which can be used directly for CNN training and evaluation.

How to use the interpolation script which fills in the missing information for frequency channels that are masked due to RFI \
python interpolate_chime_frbs.py --input-dir /path/to/chime_h5_files --output chime_interp 
