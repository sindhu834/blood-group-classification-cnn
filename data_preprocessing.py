import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def normalize_image(image):
    """ Normalize an image by scaling pixel values to [0, 1] range. """
    return image.astype('float32') / 255.0


def augment_image(image):
    """ Perform augmentation on an image. """
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    # Reshape image to include an additional dimension for augmentation
    image = np.expand_dims(image, axis=0)
    return next(datagen.flow(image, batch_size=1))


# Example usage:
# image = cv2.imread('path_to_image.jpg')
# normalized_image = normalize_image(image)
# augmented_image = augment_image(normalized_image)