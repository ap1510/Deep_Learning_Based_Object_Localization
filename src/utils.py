import os
import cv2
import numpy as np
import pickle

def load_pickle(file_path):
    """Loads a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, file_path):
    """Saves data to a pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_images_from_folder(folder, img_size=28):
    """Loads and resizes images from a folder."""
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
    return np.array(images)

def normalize_images(images):
    """Normalizes pixel values of images to range [0,1]."""
    return images / 255.0

def augment_image(image):
    """Applies random transformations to an image."""
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), np.random.randint(-20, 20), 1)
    return cv2.warpAffine(image, M, (cols, rows))
