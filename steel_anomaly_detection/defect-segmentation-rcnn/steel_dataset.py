import mrcnn.utils as utils
import numpy as np
import pandas as pd
from scipy.ndimage import label as scipy_label
from scipy.ndimage import generate_binary_structure

class SteelDataset(utils.Dataset):
    def load_steel(self, dataset_dir, files):
        """Load the steel dataset."""
        # Add defect classes
        self.add_class("steel", 1, "defect1")
        self.add_class("steel", 2, "defect2")
        self.add_class("steel", 3, "defect3")
        self.add_class("steel", 4, "defect4")

        # Load annotations CSV
        annotations = pd.read_csv(os.path.join(dataset_dir, 'train.csv'))

        # Filter rows with non-null EncodedPixels
        annotations = annotations[annotations['EncodedPixels'].notna()].copy()

        # Split ImageId_ClassId column
        annotations[['ImageId', 'ClassId']] = annotations['ImageId_ClassId'].str.split("_", expand=True)

        for file in files:
            encoded_pixels = annotations['EncodedPixels'][annotations['ImageId'] == file].tolist()
            class_ids = annotations['ClassId'][annotations['ImageId'] == file].astype(int).tolist()

            self.add_image(
                source="steel",
                image_id=file,
                path=os.path.join(dataset_dir, 'train_images', file),
                rle=encoded_pixels,
                classes=class_ids,
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image."""
        image_info = self.image_info[image_id]
        if image_info["source"] != "steel":
            return super(self.__class__, self).load_mask(image_id)

        rle_list = image_info['rle']
        class_ids = image_info['classes']
        height, width = 256, 1600  # Dimensions based on the dataset
        mask = np.zeros((height, width, len(rle_list)), dtype=np.uint8)

        # Decode RLE masks
        for i, rle in enumerate(rle_list):
            if rle:  # Only process non-empty RLEs
                rle_array = [int(x) for x in rle.split()]
                starts = rle_array[0::2] - 1
                lengths = rle_array[1::2]
                for start, length in zip(starts, lengths):
                    mask.flat[start : start + length] = 1

        return mask.astype(bool), np.array(class_ids, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "steel":
            return info["path"]
        return super(self.__class__, self).image_reference(image_id)
