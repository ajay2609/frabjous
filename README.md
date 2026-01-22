# frabjous

**FRABJOUS** is a deep learning–based framework for classifying distinct Fast Radio Burst (FRB) morphologies.

The framework provides tools to simulate FRB dynamic spectra, preprocess real data, and perform inference using either binary (one-vs-one) or multi-class convolutional neural network (CNN) models.

The repository includes scripts and configuration files used to simulate FRB dynamic spectra of different morphological classes. These simulations are used to generate labelled datasets for training the CNN models.

## Using the Simulation Scripts
1. Go to the simulation directory (frabjous_sim)
```bash
cd frabjous_sim
```
2. Edit the YAML configuration file corresponding to the desired morphology to set parameter ranges (e.g. DM, scattering, drift rate, S/N, number of bursts).
3. Run the simulation for a particular archetype 
```bash
python simulate_<morphology_type>.py
```
4. The script generates simulated FRB dynamic spectra along with metadata, which can be used directly for CNN training and evaluation.

## Inerpolating masked channels
In general data from radio telescope is affected with RFI, particularly presence of narrowband RFI leads to masking of frequency channels. For testing with first CHIME/FRB catalog we developed our interpolation method to fill in the missing in the missing information. This can also be used to for interpolation for interpolation for data with other telescopes at different frequencies.
Here is in example of how to run the interpolation script for chime FRBs 
```bash
python interpolate_chime_frbs.py --input-dir /path/to/chime_h5_files --output chime_interp 
```
The script outputs interpolated FRB dynamic spectra in numpy format, which can be used directly in the inference pipelines.

## Inference using a set of binary models
here each model distinguishes between a pair of FRB morphological classes.
```bash
python inference/run_binary_inference.py \
  --data chime_interp_frbs.npz \
  --models-dir models/binary \
  --output-dir results/
```
Outputs : all\_scores.npy: confidence scores for each morphological class for every input FRB and All\_scores.txt: final predicted class for each FRB.

## Inference Using a Multi-Class Model

As an alternative to classification with a set of binary models, inference can be run with single mutli-class classifier which gives confidence for all archetypes.
```bash
python inference/run_binary_inference.py \
  --data chime_interp_frbs.npz \
  --models-dir models/ \
  --output-dir results/
```
## Examples 

The repository includes example training scripts and configuration files for training both binary and multi-class CNN models. These scripts allow retraining models with custom simulation parameters and alternative network architectures.
[typeIIvsIV](notebooks/classify_IIvsIV.ipynb)
Details are provided within the corresponding training scripts.```
