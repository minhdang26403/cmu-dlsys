import gzip
from typing import List, Optional

import numpy as np

from ..data_basic import Dataset


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)

        with gzip.open(image_filename, "rb") as f:
            # Read the header info: magic number, number of images, rows, and columns.
            # The '>I' dtype specifies big-endian unsigned 4-byte integers.
            _, num_images, num_rows, num_cols = np.frombuffer(f.read(16), dtype=">I")

            # Read the rest of the file, which contains all the raw pixel data.
            pixel_data = np.frombuffer(f.read(), dtype=np.uint8)

            # Reshape the 1D pixel array into a 2D matrix.
            # Each row corresponds to a flattened 28x28 image (784 pixels).
            self.X = pixel_data.reshape(num_images, num_rows * num_cols)

            # Normalize pixel values to the [0.0, 1.0] range.
            self.X = np.divide(self.X, 255, dtype=np.float32)

        with gzip.open(label_filename, "rb") as f:
            # Read and discard the 8-byte header (magic number, number of labels).
            _, num_labels = np.frombuffer(f.read(8), dtype=">I")

            # Read the rest of the file, which contains the label for each image.
            self.y = np.frombuffer(f.read(), dtype=np.uint8)

        assert num_images == num_labels, "Number of images and labels must match"
        self.num_samples = num_images
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        image = self.X[index]
        label = self.y[index]
        if self.transforms is not None:
            # Since we flattened the images, we need to reshape them back to 28x28x1 for
            # the transforms to work.
            image = self.apply_transforms(image.reshape(28, 28, 1)).flatten()
        return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.num_samples
        ### END YOUR SOLUTION
