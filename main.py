import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = tf.keras.models.load_model("models/chest_xray_model.h5")

labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening',
                'Hernia']

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)[0]
    results = {label: round(pred, 2) for label, pred in zip(labels, prediction)}
    return results

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk


        predictions = predict_image(file_path)
        result_text.set("Predictions:\n" + "\n".join([f"{label}: {confidence:.2f}" for label, confidence in predictions.items()]))

window = tk.Tk()
window.title("Chest X-Ray Evaluation")
window.geometry("900x900")


Label(window, text="Chest X-Ray Evaluation Tool", font=("Helvetica", 16)).pack(pady=10)
img_label = Label(window)
img_label.pack(pady=10)
Button(window, text="Upload X-Ray Image", command=upload_image).pack(pady=10)
result_text = tk.StringVar()
result_text.set("Upload an image to analyze.")
Label(window, textvariable=result_text, font=("Helvetica", 12)).pack(pady=10)

window.mainloop()
