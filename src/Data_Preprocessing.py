import os
import cv2
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Define constants
TRAIN_DIR = "D:\flipkart grid\images"
IMG_SIZE = 28
LABEL_FILE = "training.csv"

# Load labels
def load_labels():
    df = pd.read_csv(LABEL_FILE)
    df = df.set_index("image_name")  # Ensure image names are the index
    return df

# Process images
def process_images(label_data, limit=800):
    x, y = [], []
    for i, img_name in tqdm(enumerate(os.listdir(TRAIN_DIR)), total=limit):
        if i >= limit:
            break
        path = os.path.join(TRAIN_DIR, img_name)
        if os.path.isfile(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            x.append(img)
            if img_name in label_data.index:
                y.append(label_data.loc[img_name].values)
    return np.array(x), np.array(y)

# Save processed data
def save_data(x, y, test_size=0.1):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, stratify=y)
    with open('train_x.pickle', 'wb') as f: pickle.dump(x_train, f)
    with open('train_y.pickle', 'wb') as f: pickle.dump(y_train, f)
    with open('test_x.pickle', 'wb') as f: pickle.dump(x_test, f)
    with open('test_y.pickle', 'wb') as f: pickle.dump(y_test, f)

# Execute data processing
if __name__ == "__main__":
    labels = load_labels()
    x_data, y_data = process_images(labels)
    save_data(x_data, y_data)
    print("Data processing complete. Train and test datasets saved.")
