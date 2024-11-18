from mrcnn.config import Config

class SteelConfig(Config):
    """Configuration for training on the steel defect detection dataset."""
    
    # Give the configuration a recognizable name
    NAME = "steel"

    # System settings
    GPU_COUNT = 1  # Use one GPU
    IMAGES_PER_GPU = 2  # Images per GPU (Batch size = GPU_COUNT * IMAGES_PER_GPU)

    # Dataset specifications
    NUM_CLASSES = 1 + 4  # Background + 4 defect classes

    # Input image resizing
    IMAGE_MIN_DIM = 256  # Increased from 128 for better resolution
    IMAGE_MAX_DIM = 256

    # Anchor configurations
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # Slightly larger for better coverage
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]  # Standard aspect ratios

    # ROI and detection settings
    TRAIN_ROIS_PER_IMAGE = 32  # Training regions of interest
    DETECTION_MIN_CONFIDENCE = 0.7  # Lowered slightly for better recall
    DETECTION_NMS_THRESHOLD = 0.3  # Suppress overlapping detections

    # RPN proposal filtering
    RPN_NMS_THRESHOLD = 0.7  # Suppress duplicate region proposals during training

    # Training settings
    STEPS_PER_EPOCH = 500  # Adjusted based on dataset size
    VALIDATION_STEPS = 50  # Increased for more stable validation metrics

    # Epochs
    EPOCHS = 50  # Lowered to prevent overfitting during initial training
    
    # Learning rate settings
    LEARNING_RATE = 0.001  # Default learning rate
    LEARNING_MOMENTUM = 0.9  # Default momentum for SGD
    
    # Augmentation
    AUGMENT = True  # Placeholder for augmentation configuration
    
    # Optimizer settings
    WEIGHT_DECAY = 0.0001  # Regularization

    # Mask settings
    USE_MINI_MASK = True  # Use smaller masks to save memory
    MINI_MASK_SHAPE = (56, 56)  # Shape of the mini mask

    # Debugging and display
    DETECTION_MAX_INSTANCES = 100  # Max number of instances detected in an image
    BACKBONE = "resnet50"  # Backbone architecture (ResNet-50 is faster)