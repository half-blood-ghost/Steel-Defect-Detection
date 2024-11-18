import numpy as np

def dice(predict, label):
    """
    Calculate the Dice coefficient for predictions and ground truth.

    Parameters:
        predict (np.ndarray): Predicted output, shape (5, 256, 1600), with 5 classes (including background).
        label (np.ndarray): Ground-truth mask, shape (4, 256, 1600), with 4 defect classes.

    Returns:
        pos (float): Dice coefficient for defect classes (1-4).
        neg (float): Dice coefficient for the background class (0).
    """
    # Find the predicted class for each pixel
    predict_classes = predict.argmax(axis=0)

    # Combine ground-truth masks into a single array with class labels
    mask = np.zeros((256, 1600), dtype=int)
    for i in range(label.shape[0]):  # Loop through the 4 defect classes
        mask += label[i] * (i + 1)

    # Dice score for the background class (0)
    background_pred = (predict_classes == 0).astype(int)
    background_true = (mask == 0).astype(int)
    intersection_neg = np.sum(background_pred * background_true)
    total_neg = np.sum(background_pred) + np.sum(background_true)
    neg = 1 if total_neg == 0 else (2 * intersection_neg / total_neg)

    # Dice score for defect classes (1-4)
    intersection_pos = 0
    total_pos = 0
    for i in range(1, 5):  # Loop through defect classes
        class_pred = (predict_classes == i).astype(int)
        class_true = (mask == i).astype(int)
        intersection_pos += np.sum(class_pred * class_true)
        total_pos += np.sum(class_pred) + np.sum(class_true)

    pos = 1 if total_pos == 0 else (2 * intersection_pos / total_pos)

    return pos, neg