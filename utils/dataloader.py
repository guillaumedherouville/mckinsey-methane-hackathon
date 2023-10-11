import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import rasterio
import pandas as pd


class LoadData:
    def __init__(self,
                metadata_path="methane-hackathon/data/train_data/metadata.csv",
                image_data_path="methane-hackathon/data/train_data/"):
        self.metadata_path = metadata_path
        self.image_data_path = image_data_path

    
    def get_train_data(self):
        
        image_data = []
        plume_labels = []

        meta_df = pd.read_csv(self.metadata_path)
        meta_df["path"] = meta_df['path'].astype(str) + '.tif'
        # Loop through the metadata and load images
        for index, row in meta_df.iterrows():
            image_path = row['path']
            plume_label = row['plume']

            # Read the TIFF image using rasterio
            try:
                with rasterio.open(self.image_data_path+image_path) as src:
                    image = src.read(1)  # Assuming single-band image, adjust if necessary
                    # You may want to resize or preprocess the image here if necessary

                # Append the image data and plume label to their respective lists
                image_data.append(image)
                plume_labels.append(plume_label)
            except Exception as e:
                print(f"Error loading image at {image_path}: {e}")

        # Convert the lists into NumPy arrays
        image_data = np.array(image_data)
        plume_labels = np.array([1 if i=="yes" else 0 for i in plume_labels])
        
        return image_data, plume_labels
    
    
    def augment_data(self, X_train, y_train, n_epochs):
        datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
        batch_size = 32
        steps_per_epoch = len(X_train) // batch_size
        X_train = X_train.reshape(-1,64,64,1)
        # Fit the data augmentation generator to your training data
        datagen.fit(X_train)
        augmented_images = []
        augmented_labels = []
        # Generate augmented images and labels
        for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=batch_size,shuffle=True):
            augmented_images.append(X_batch)
            augmented_labels.append(y_batch)
            if len(augmented_images) >= len(X_train):
                break
        X_train_augmented = np.concatenate(augmented_images)
        y_train_augmented = np.concatenate(augmented_labels)
        return X_train_augmented, y_train_augmented

    
    def normalize_data(self, image_data):
        min_val = image_data.min()
        max_val = image_data.max()
        return (image_data - min_val) / (max_val - min_val)
