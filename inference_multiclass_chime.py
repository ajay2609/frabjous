#!/usr/bin/env python3
"""
Inference script for multi-class FRB classification on CHIME data.

This script loads a trained multi-class classification model and uses it to
make predictions on CHIME FRB data, outputting confidence scores for each class.

Usage:
    python inference_multiclass_chime.py [--model-dir PATH] [--data-file PATH] 
                                          [--label-file PATH] [--output-file PATH]

The script performs the following steps:
    1. Loads the trained model from the specified directory (default: models/model_single_multi/)
    2. Loads CHIME testing data:
       - NumPy .npz file containing FRB images (default: files/chime_interp_frbs.npz)
       - Text file containing morphology labels (default: files/check.txt.txt)
    3. Uses the model to generate predictions on the CHIME testing data
    4. Extracts confidence scores for each class (A, B, C, C1, C2)
    5. Outputs the confidence scores to a text file in tabular format (default: confidence_scores.txt)

Output format:
    The output file contains tab-separated values with the following columns:
    - Index: Sample index (0-based)
    - True_Label: Ground truth label from the label file
    - A, B, C, C1, C2: Confidence scores (probabilities) for each class
    - Predicted_Class: The class with the highest confidence score

Example output:
    Index   True_Label  A       B       C       C1      C2      Predicted_Class
    0       A           0.9234  0.0456  0.0123  0.0098  0.0089  A
    1       B           0.0345  0.8765  0.0234  0.0456  0.0200  B
"""

import logging
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_path):
    """
    Load a trained Keras model.
    
    Parameters
    ----------
    model_path : str or Path
        Path to the directory containing the saved model.
    
    Returns
    -------
    model : keras.Model
        The loaded Keras model.
    """
    try:
        model = keras.models.load_model(str(model_path))
        logger.info(f"Successfully loaded model from: {model_path}")
        return model
    except Exception as exc:
        logger.error(f"Failed to load model from {model_path}: {exc}")
        raise


def load_chime_data(data_file, label_file):
    """
    Load CHIME FRB images and their labels.
    
    Parameters
    ----------
    data_file : str or Path
        Path to the NumPy .npz file containing FRB images.
    label_file : str or Path
        Path to a text file containing morphology labels (one per line: A, B, C, C1, or C2).
    
    Returns
    -------
    images : np.ndarray
        Normalized FRB images with shape (n_samples, height, width, 1).
    labels : list of str
        Morphology labels for each image.
    """
    # Load images
    with np.load(data_file) as frbdata:
        frbd = frbdata["arr_0"]
    
    # Normalize images
    normalized_images = []
    for img in frbd:
        immax = img.max()
        if immax > 0:
            norm_img = img / (immax / 255.0)
        else:
            norm_img = img
        normalized_images.append(norm_img)
    
    images = np.array(normalized_images)
    
    # Add channel dimension if not present
    if images.ndim == 3:
        images = np.expand_dims(images, axis=-1)
    
    logger.info(f"Loaded {len(images)} images with shape: {images.shape}")
    
    # Load labels
    with open(label_file, 'r') as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    
    logger.info(f"Loaded {len(labels)} labels")
    
    if len(images) != len(labels):
        logger.warning(f"Mismatch: {len(images)} images but {len(labels)} labels")
    
    return images, labels


def predict_with_confidence(model, images):
    """
    Generate predictions and extract confidence scores.
    
    Parameters
    ----------
    model : keras.Model
        The trained classification model.
    images : np.ndarray
        Input images with shape (n_samples, height, width, 1).
    
    Returns
    -------
    predictions : np.ndarray
        Predicted class indices with shape (n_samples,).
    confidences : np.ndarray
        Confidence scores for each class with shape (n_samples, n_classes).
    """
    logger.info("Generating predictions...")
    
    # Get model predictions (softmax probabilities)
    confidences = model.predict(images)
    
    # Get predicted class for each sample
    predictions = np.argmax(confidences, axis=1)
    
    logger.info(f"Predictions complete. Shape: {confidences.shape}")
    
    return predictions, confidences


def save_confidence_scores(confidences, labels, output_file, class_names=None):
    """
    Save confidence scores to a text file in tabular format.
    
    Parameters
    ----------
    confidences : np.ndarray
        Confidence scores with shape (n_samples, n_classes).
    labels : list of str
        True labels for each sample.
    output_file : str or Path
        Path to the output text file.
    class_names : list of str, optional
        Names for each class. Defaults to ['A', 'B', 'C', 'C1', 'C2'].
    """
    if class_names is None:
        class_names = ['A', 'B', 'C', 'C1', 'C2']
    
    output_path = Path(output_file)
    
    with open(output_path, 'w') as f:
        # Write header
        header = "Index\tTrue_Label\t" + "\t".join(class_names) + "\tPredicted_Class\n"
        f.write(header)
        
        # Write data for each sample
        for i, (conf, true_label) in enumerate(zip(confidences, labels)):
            pred_class = class_names[np.argmax(conf)]
            
            # Format confidence scores to 4 decimal places
            conf_str = "\t".join([f"{score:.4f}" for score in conf])
            
            line = f"{i}\t{true_label}\t{conf_str}\t{pred_class}\n"
            f.write(line)
    
    logger.info(f"Confidence scores saved to: {output_path}")


def main():
    """Main function to run the inference pipeline."""
    parser = argparse.ArgumentParser(
        description="Run inference on CHIME data with a trained multi-class FRB classifier"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/model_single_multi",
        help="Path to the directory containing the trained model (default: models/model_single_multi)"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="files/chime_interp_frbs.npz",
        help="Path to the CHIME data .npz file (default: files/chime_interp_frbs.npz)"
    )
    parser.add_argument(
        "--label-file",
        type=str,
        default="files/check.txt.txt",
        help="Path to the label text file (default: files/check.txt.txt)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="confidence_scores.txt",
        help="Path to the output confidence scores file (default: confidence_scores.txt)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    model_dir = Path(args.model_dir)
    data_file = Path(args.data_file)
    label_file = Path(args.label_file)
    output_file = Path(args.output_file)
    
    # Validate input files exist
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        return 1
    
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return 1
    
    if not label_file.exists():
        logger.error(f"Label file not found: {label_file}")
        return 1
    
    # Load model
    logger.info("=" * 60)
    logger.info("STEP 1: Loading trained model")
    logger.info("=" * 60)
    model = load_model(model_dir)
    
    # Load data
    logger.info("=" * 60)
    logger.info("STEP 2: Loading CHIME test data")
    logger.info("=" * 60)
    images, labels = load_chime_data(data_file, label_file)
    
    # Generate predictions
    logger.info("=" * 60)
    logger.info("STEP 3: Generating predictions")
    logger.info("=" * 60)
    predictions, confidences = predict_with_confidence(model, images)
    
    # Save results
    logger.info("=" * 60)
    logger.info("STEP 4: Saving confidence scores")
    logger.info("=" * 60)
    save_confidence_scores(confidences, labels, output_file)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples processed: {len(images)}")
    logger.info(f"Model input shape: {model.input_shape}")
    logger.info(f"Model output shape: {model.output_shape}")
    logger.info(f"Confidence scores shape: {confidences.shape}")
    logger.info(f"Results saved to: {output_file}")
    logger.info("=" * 60)
    logger.info("Inference complete!")
    
    return 0


if __name__ == "__main__":
    exit(main())
