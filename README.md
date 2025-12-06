# Casting Defect Detection (TensorFlow)

*A small CNN project to classify aluminium casting images as OK or defective.*

## 👋 About this project

This project is part of my learning path as a **junior data scientist**.  
I wanted to build something close to a **real industrial use case**: automatic visual inspection in manufacturing.

The goal:

> Given a grayscale image of an aluminium casting (front view), classify it as  
> **`def_front`** (defective) or **`ok_front`** (non-defective).

The notebook is written to be easy to follow and reusable as a student / junior GitHub project.   

---

## 🔍 What this project demonstrates

From a junior DS point of view, this repo shows that I can:

- Organise a project with a clear **step-by-step pipeline**
- Use **TensorFlow / Keras** to build and train a **Convolutional Neural Network (CNN)**
- Keep preprocessing consistent between training and inference (grayscale, resize, rescaling)
- Split data into **train / validation / test** sets
- Monitor **training curves** (loss & accuracy)
- Evaluate using:
  - Test accuracy & loss
  - **Confusion matrix**
  - **Classification report** (precision, recall, F1-score)
- Write a helper to **predict a single image** and visualise the result
- Save the trained model so it can be reloaded without retraining

The notebook structure reflects this:

1. Imports and settings  
2. Data folders  
3. Quick look at the data (sample images)  
4. TensorFlow datasets (`image_dataset_from_directory`)  
5. Build a small CNN  
6. Train the model  
7. Evaluate on validation and test data  
8. Predict a single image  
9. Save the trained model   

---

## 🧠 Model & approach (high level)

**Data & preprocessing**

- Input: grayscale images of cast parts (two folders: `def_front` and `ok_front`)  
- Images are resized to **128×128** and kept in one channel.  
- A Keras `Rescaling(1./255)` layer normalises pixel values to \[0, 1].   

**Model**

The CNN is intentionally small so it can train quickly on a laptop:

- Optional `RandomFlip` + `RandomRotation` data augmentation
- Several blocks of:
  - `Conv2D` with ReLU activation
  - `MaxPooling2D`
- `Flatten`
- `Dense` layer(s) with `Dropout` for regularisation
- Final `Dense(1, activation="sigmoid")` for binary classification   

**Training**

- Loss: `BinaryCrossentropy`
- Optimiser: `Adam`
- Metric: `Accuracy`
- Early stopping on validation loss to avoid overfitting

**Single-image prediction**

A function `predict_image(path, model, class_names)`:

- Loads a single image, converts to grayscale and resizes to `IMG_SIZE`
- Applies the same preprocessing as the training pipeline
- Runs `model.predict` and displays the image with predicted class and defect probability   

**Saving the model**

- The trained model is saved to a `models/` folder so it can be reused later without retraining.

---

## 📊 Results (current version)

On the final trained model:

- **Validation accuracy:** ~**0.95**  
- **Test accuracy:** ~**0.96**   
- The confusion matrix and classification report show **balanced performance** on both classes (`def_front` and `ok_front`), with precision and recall both around **0.93–0.96**.   

In plain words: the model is able to correctly distinguish defective vs OK parts most of the time, which would already be useful as a first quality-control assistant.
