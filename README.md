# frabjous

This is a deep learning-based framework to classify distinct FRB morphologies.

The repository also includes frabjous_sim, which contains scripts and configuration files used to simulate FRB dynamic spectra of different morphological classes. These simulations are used to generate labelled datasets for training and testing the CNN models.

## Using the Simulation Scripts
1. Go to the simulation directory (frabjous_sim)
2. Edit the YAML configuration file corresponding to the desired morphology to set parameter ranges (e.g. DM, scattering, drift rate, S/N, number of bursts).
3. python <simulation_script>.py
4. The script generates simulated FRB dynamic spectra along with metadata, which can be used directly for CNN training and evaluation.

## Data Preprocessing

### Interpolating CHIME Data
Use the interpolation script to fill in missing information for frequency channels that are masked due to RFI:

```bash
python interpolate_chime_frbs.py --input-dir /path/to/chime_h5_files --output chime_interp
```

## Model Inference

### Multi-Class Classification on CHIME Data
Use the inference script to make predictions on CHIME FRB data with a trained multi-class model:

```bash
python inference_multiclass_chime.py --model-dir models/model_single_multi \
                                      --data-file files/chime_interp_frbs.npz \
                                      --label-file files/check.txt.txt \
                                      --output-file confidence_scores.txt
```

**Input Requirements:**
- `--model-dir`: Directory containing the trained Keras model (default: `models/model_single_multi`)
- `--data-file`: NumPy `.npz` file with FRB images (default: `files/chime_interp_frbs.npz`)
- `--label-file`: Text file with morphology labels, one per line (A, B, C, C1, or C2) (default: `files/check.txt.txt`)
- `--output-file`: Output file for confidence scores (default: `confidence_scores.txt`)

**Output Format:**
The script generates a tab-separated file with the following columns:
- `Index`: Sample index (0-based)
- `True_Label`: Ground truth label from the label file
- `A`, `B`, `C`, `C1`, `C2`: Confidence scores (probabilities) for each morphology class
- `Predicted_Class`: The class with the highest confidence score

**Example Output:**
```
Index   True_Label  A       B       C       C1      C2      Predicted_Class
0       A           0.9234  0.0456  0.0123  0.0098  0.0089  A
1       B           0.0345  0.8765  0.0234  0.0456  0.0200  B
2       C           0.0123  0.0234  0.8543  0.0567  0.0533  C
```

### Binary Classification on CHIME Data
For binary classification using multiple models:

```bash
python inference_with_binary_models.py
```

This script uses environment variables `DATA_DIR`, `MODELS_DIR`, and `OUT_DIR` to specify paths.
