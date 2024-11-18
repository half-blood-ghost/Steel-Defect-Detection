import numpy as np

def convert_for_dice(masks, class_ids):
    """
    Converts masks and class IDs into a prediction array with logical OR operations.

    Args:
        masks (np.ndarray): Binary masks of shape (H, W, num_classes).
        class_ids (list or np.ndarray): List of class indices for each mask.

    Returns:
        np.ndarray: Prediction array of shape (5, H, W).
    """
    predict = np.zeros((5, masks.shape[0], masks.shape[1]))
    predict[0, :, :] = 0.1  # Initialize background class with a small value.
    
    for i, class_index in enumerate(class_ids):
        predict[class_index, :, :] = np.logical_or(predict[class_index, :, :], masks[:, :, i])
    
    return predict

def dice(predict, label):
    """
    Computes the Dice coefficient for predictions and ground truth labels.

    Args:
        predict (np.ndarray): Predicted array of shape (5, H, W).
        label (np.ndarray): Ground truth array of shape (4, H, W).

    Returns:
        tuple: Dice scores for positive classes and the background class.
    """
    # Convert predictions to class indices.
    predict = predict.argmax(axis=0)

    # Combine individual masks into a single mask with class labels.
    mask = np.zeros_like(predict)
    for i in range(label.shape[0]):
        mask += label[i, :, :] * (i + 1)

    # Initialize total intersection and union counters.
    total_intersection_neg = total_sum_neg = 0
    total_intersection_pos = total_sum_pos = 0

    # Compute Dice score for background class (class 0).
    m1 = (predict == 0).astype(int)
    m2 = (mask == 0).astype(int)
    total_intersection_neg = np.sum(m1 * m2)
    total_sum_neg = np.sum(m1) + np.sum(m2)
    neg = 1 if total_intersection_neg == 0 else 2 * total_intersection_neg / total_sum_neg

    # Compute Dice scores for positive classes (classes 1-4).
    for i in range(1, 5):
        m1 = (predict == i).astype(int)
        m2 = (mask == i).astype(int)
        intersection = np.sum(m1 * m2)
        total_intersection_pos += intersection
        total_sum_pos += np.sum(m1) + np.sum(m2)

    pos = 1 if total_sum_pos == 0 else 2 * total_intersection_pos / total_sum_pos

    return pos, neg

def get_score(masks, class_ids, r):
    """
    Computes the Dice score for given masks and predictions.

    Args:
        masks (np.ndarray): Ground truth masks of shape (H, W, num_classes).
        class_ids (list or np.ndarray): Class indices for ground truth masks.
        r (dict): Predicted masks and class IDs (keys: 'masks', 'class_ids').

    Returns:
        tuple: Dice scores for positive classes and the background class.
    """
    # Convert ground truth to prediction format.
    label = convert_for_dice(masks, class_ids)[1:]  # Exclude background class.
    
    # Convert predictions to the same format.
    predict = convert_for_dice(r['masks'], r['class_ids'])
    
    # Compute the Dice score.
    return dice(predict, label)
