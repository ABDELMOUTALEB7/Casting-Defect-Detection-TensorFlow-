# Casting Defect Detection with CNN  
*A junior data science project on computer vision & quality control*

## 👋 About this project

I built this project as a **junior data science / ML profile** to show that I can:

- Work with a **real industrial dataset**
- Build and train a **Convolutional Neural Network (CNN)** with TensorFlow/Keras
- Evaluate a model properly (not only accuracy)
- Explain results and limits in a clear, structured way

The goal is simple:

> Given a grayscale image of a cast part, classify it as  
> **`def_front` (defective)** or **`ok_front` (non-defective)**.

This type of problem is very common in **manufacturing quality control**: automatically detect defects and reduce manual inspection time.

---

## 🎯 What this project demonstrates

From a junior data scientist point of view, this repo shows that I can:

- Set up a **train / validation / test split** and keep it consistent
- Build a CNN from scratch in Keras (no pre-trained model)
- Use **Rescaling** layers and keep preprocessing consistent between training and inference
- Track and interpret:
  - Training & validation **loss / accuracy curves**
  - **Confusion matrix** & **classification report**
  - **ROC** and **Precision–Recall** curves
- Inspect **misclassified images** to understand model weaknesses
- Implement a small utility to **predict a single image**
- Save and reload a trained model

---

## 🧠 Model & approach (high level)

1. **Data loading**
   - Grayscale images of cast parts (front view)
   - Split into **train**, **validation**, and **test** sets

2. **Preprocessing**
   - Resize images to a fixed size (e.g. `128x128`)
   - Use a Keras `Rescaling(1./255)` layer to normalise pixels to \[0, 1\]

3. **CNN architecture (TensorFlow / Keras)**
   - Several blocks of:
     - `Conv2D` + `ReLU` activation
     - `MaxPooling2D`
   - `Flatten` layer
   - `Dense` layers with `Dropout` for regularisation
   - Final `Dense(1, activation="sigmoid")` for binary classification

4. **Training**
   - Loss: `BinaryCrossentropy`
   - Optimiser: `Adam`
   - Metrics: `Accuracy`
   - Early stopping / checkpoints can be added (depending on version)

5. **Evaluation & analysis**
   - Accuracy & loss on **validation** and **test** sets
   - **Confusion matrix** & **classification report** for both classes
   - **ROC curve** & **Precision–Recall curve**
   - Visualisation of some **misclassified images**
   - A simple **decision threshold exploration** to see the trade-off between false positives and false negatives

6. **Single-image prediction**
   - A helper function `predict_image(path, model, class_names)` that:
     - Loads one image
     - Applies the same preprocessing as training
     - Displays it with the predicted label and probability

7. **Model saving**
   - Save the trained model as a Keras file (e.g. `models/defect_detection_cnn.keras`)  
   - Can be loaded later for inference without retraining

---

## 📊 Current results (summary)

> Exact numbers may change slightly when retraining, but this is the typical performance.

- **Validation accuracy**: ~0.86  
- **Test accuracy**: ~0.85  
- The model successfully captures a large part of the signal needed to distinguish `def_front` vs `ok_front`.
- Confusion matrix and ROC/PR curves show that:
  - Defective parts are usually detected correctly
  - There are some false positives / false negatives, which I analyse in the notebook

The notebook also includes a short discussion about what these errors mean in a **real factory context** (cost of missing a defect vs cost of false alarms).
