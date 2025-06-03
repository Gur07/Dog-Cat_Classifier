# 🐶🐱 Cat vs Dog Classifier (CNN from Scratch)

A Convolutional Neural Network built from scratch using TensorFlow and Keras to classify images as either a **cat** or a **dog**.

## 🧠 Project Overview

This project implements a simple yet effective CNN architecture trained on a labeled dataset of cat and dog images. It achieved an accuracy of **92%** on the validation set.

## 🛠️ Technologies Used
- Python
- TensorFlow & Keras
- OpenCV
- NumPy, Matplotlib

## 📂 Architecture

```python
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

📊 Results
Validation Accuracy: 92%

Loss: Stable convergence after ~15 epochs

📚 Key Learnings
Built custom CNN with batch normalization and dropout

Understood importance of kernel size and pooling layers

Explored regularization techniques to avoid overfitting
