import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

BASE_DIR = r"C:\Users\sidc2\.cache\kagglehub\datasets\nih-chest-xrays\data\versions\3"
IMAGE_FOLDERS = [os.path.join(BASE_DIR, f"images_{str(i).zfill(3)}\\images") for i in range(1, 13)]
CSV_FILE = os.path.join(BASE_DIR, "Data_Entry_2017.csv")
TRAIN_VAL_LIST = os.path.join(BASE_DIR, "train_val_list.txt")
TEST_LIST = os.path.join(BASE_DIR, "test_list.txt")

image_paths = {}
for folder in IMAGE_FOLDERS:
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                image_paths[file] = os.path.join(root, file)

df = pd.read_csv(CSV_FILE)
df["Path"] = df["Image Index"].apply(lambda x: image_paths.get(x, None))
df = df.dropna(subset=["Path"])

df["Labels"] = df["Finding Labels"].str.split('|')

mlb = MultiLabelBinarizer()
label_binarized = mlb.fit_transform(df["Labels"])

for idx, label in enumerate(mlb.classes_):
    df[label] = label_binarized[:, idx]

with open(TRAIN_VAL_LIST, "r") as f:
    train_val_files = set(f.read().splitlines())
with open(TEST_LIST, "r") as f:
    test_files = set(f.read().splitlines())

train_val_df = df[df["Image Index"].isin(train_val_files)].copy()
test_df = df[df["Image Index"].isin(test_files)].copy()

train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1.0/255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col="Path",
    y_col=mlb.classes_,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw"
)
val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col="Path",
    y_col=mlb.classes_,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw"
)
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col="Path",
    y_col=mlb.classes_,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw"
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(mlb.classes_), activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint("chest_xray_model.keras", save_best_only=True, monitor="val_loss", mode="min")
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[checkpoint, early_stopping]
)

test_loss, test_accuracy = model.evaluate(test_generator)
model.save("chest_xray_model.keras")
model.save("chest_xray_model.h5", save_format="h5")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")