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

## Interpolating masked channels
In general data from radio telescope is affected with RFI, particularly presence of narrowband RFI leads to masking of frequency channels. For testing with first CHIME/FRB catalog we developed our interpolation method to fill in the missing in the missing information. This can also be used to for interpolation for interpolation for data with other telescopes at different frequencies.
Here is in example of how to run the interpolation script for chime FRBs. Before that download h5 files for all the bursts from first CHIME/FRB catalog.  
```bash
python interpolate_chime_frbs.py --input-dir /path/to/chime_h5_files --output chime_interp_frbs 
```
The script outputs interpolated FRB dynamic spectra in numpy format, which can be used directly in the inference pipelines.

## Inference using a set of binary models
here each model distinguishes between a pair of FRB morphological classes.
```bash
python inference/inference_with_binary_models.py \
  --data files/chime_interp_frbs.npz \
  --models-dir models/ \
  --output-dir results/
```
Outputs : all\_scores.npy: confidence scores for each morphological class for every input FRB and All\_scores.txt: final predicted class for each FRB.

## Inference Using a Multi-Class Model

As an alternative to classification with a set of binary models, inference can be run with single mutli-class classifier which gives confidence for all archetypes.
```bash
python inference/inference_mutliclass_model.py \
  --data files/chime_interp_frbs.npz \
  --models-dir models/ \
  --output-dir results/
```
## Examples 

The repository includes example jupyter notebook for training both binary and multi-class CNN models. These notebook can be modified for retraining models with custom simulation parameters and alternative network architectures.

This [notebook](notebooks/classify_IIvsIV.ipynb) shows how to obtain a optimised model to distinguish between type II and type IV archetype and [type II vs V](notebooks/classify_IIvsV.ipynb) between type II and V archetype. The [Notebook](notebooks/single_mutliclass_model_chime.ipynb) shows how to obtain a optimised models for mutli-class classification for five types i.e type I, II, II, IV and V.

This [notebook](cm_after_inferences.ipynb) shows the confusion matrix after running inference for the bursts in the first CHIME/FRB catalog using set of binary models and single multi-class classifier. 

Details are provided within the corresponding jupyter notebooks. 
