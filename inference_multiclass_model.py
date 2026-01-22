import argparse
import numpy as np
import tensorflow as tf


def read_chime_data(data_file, label_file):
    """
    Load CHIME FRB images and correct labels.

    This function reads interpolated CHIME FRB dynamic spectra and their
    corresponding labels, normalizes each image, and splits
    the data into training and test subsets by class.

    Parameters
    ----------
    data_file : str
        Path to the NumPy `.npz` file containing FRB images.
    label_file : str
        Path to a text file containing morphology labels
        (one per line: A, B, C, C1, or C2).

    Returns
    -------
    labels_train : list
        One-hot encoded labels for training data.
    images_train : list
        Normalized FRB images for training.
    images_test : list
        Normalized FRB images for testing.
    labels_test : list
        One-hot encoded labels for testing data.
    """

    # Containers for each morphology class
    frbdn1, frbdn2, frbdn3, frbdn4, frbdn5 = [], [], [], [], []
    frbdl1, frbdl2, frbdl3, frbdl4, frbdl5 = [], [], [], [], []

    # ---- Load data ----
    with np.load(data_file) as frbdata:
        frbd = frbdata["arr_0"]

    with open(label_file) as f:
        correct_type = f.readlines()

    # ---- Assign images and labels ----
    for i in range(len(correct_type)):
        immax = frbd[i].max()
        norm_img = frbd[i] / (immax / 255)

        if correct_type[i] == "A\n":
            frbdn1.append(norm_img)
            frbdl1.append([1., 0., 0., 0., 0.])

        if correct_type[i] == "B\n":
            frbdn2.append(norm_img)
            frbdl2.append([0., 1., 0., 0., 0.])

        if correct_type[i] == "C\n":
            frbdn3.append(norm_img)
            frbdl3.append([0., 0., 1., 0., 0.])

        if correct_type[i] == "C1\n":
            frbdn4.append(norm_img)
            frbdl4.append([0., 0., 0., 1., 0.])

        if correct_type[i] == "C2\n":
            frbdn5.append(norm_img)
            frbdl5.append([0., 0., 0., 0., 1.])

    # ---- Train / test split (per class) ----
    lenA  = int(len(frbdn1) / 2)
    lenB  = int(len(frbdn2) / 2)
    lenC  = int(len(frbdn3) / 2)
    lenC1 = int(len(frbdn4) / 2)
    lenC2 = int(len(frbdn5) / 2)

    images_train = (
        frbdn1[:lenA] +
        frbdn2[:lenB] +
        frbdn3[:lenC] +
        frbdn4[:lenC1] +
        frbdn5[:lenC2]
    )

    labels_train = (
        frbdl1[:lenA] +
        frbdl2[:lenB] +
        frbdl3[:lenC] +
        frbdl4[:lenC1] +
        frbdl5[:lenC2]
    )

    images_test = (
        frbdn1[lenA:] +
        frbdn2[lenB:] +
        frbdn3[lenC:] +
        frbdn4[lenC1:] +
        frbdn5[lenC2:]
    )

    labels_test = (
        frbdl1[lenA:] +
        frbdl2[lenB:] +
        frbdl3[lenC:] +
        frbdl4[lenC1:] +
        frbdl5[lenC2:]
    )

    return labels_train, images_train,  labels_test, images_test


def output_confidence_scores_to_file(predictions, output_file, class_names):
    """
    Save confidence scores to a text file.

    Parameters
    ----------
    predictions : np.ndarray
        Confidence scores for each class (output of the model's softmax layer).
    output_file : str
        The path to the file where scores will be saved.
    class_names : list
        List of class names corresponding to the confidence scores.
    """
    with open(output_file, "w") as f:
        # Write header
        f.write("Sample\t" + "\t".join(class_names) + "\n")

        # Write each sample's confidence scores
        for index, confidence_scores in enumerate(predictions):
            f.write(f"Sample_{index + 1}\t" + "\t".join(map(str, confidence_scores)) + "\n")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run predictions on CHIME data and output confidence scores.")
    parser.add_argument("--model", type=str, required=True, help="Path to the saved model directory.")
    parser.add_argument("--data", type=str, required=True, help="Path to the NumPy .npz data file containing FRB images.")
    parser.add_argument("--labels", type=str, required=True, help="Path to the text file containing morphology labels.")
    parser.add_argument("--output", type=str, default="confidence_scores.txt", help="Path to the output text file for saving confidence scores.")
    args = parser.parse_args()

    # Load the trained model
    model = tf.keras.models.load_model(args.model)

    # Load and preprocess CHIME data
    chime_labels, chime_data, chime_test_labels, chime_test = read_chime_data(args.data,args.labels)
    chime_test = np.asarray(chime_test)     
    chime_test.shape += 1,
    chime_test_labels = np.asarray(chime_test_labels)

    # Generate predictions
    predictions = model.predict(chime_test)

    # Define class names
    class_names = ['I', 'II', 'III', 'IV', 'V']

    # Output confidence scores to a file
    output_confidence_scores_to_file(predictions, args.output, class_names)
    print(f"Confidence scores have been saved to {args.output}")


if __name__ == "__main__":
    main()
