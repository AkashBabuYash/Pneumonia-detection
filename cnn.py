import numpy as np




import streamlit as st
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ReduceLROnPlateau
import os
from PIL import Image

# --- Global Variables ---
labels = ['PNEUMONIA', 'NORMAL']
img_resize = 150
st.title("Pneumonia detection deep learning tool")
st.header("using CNN algorithm")
st.markdown("""
## 🧠 About This CNN Model

This tool uses a **Convolutional Neural Network (CNN)** to detect pneumonia from chest X-ray images. Below is a breakdown of the core components and techniques used in the model:

---

### 🔍 What is a CNN?

A **Convolutional Neural Network** is a deep learning algorithm designed to process structured arrays of data such as images. CNNs automatically and adaptively learn spatial hierarchies of features from input images through the use of multiple building blocks:

- **Convolutional Layers**: These layers apply a number of filters (or kernels) to extract features such as edges, textures, and shapes from the input image.
- **ReLU Activation**: The Rectified Linear Unit (ReLU) introduces non-linearity to the model. It replaces negative values with zero: `f(x) = max(0, x)`.
- **Batch Normalization**: Speeds up training and improves stability by normalizing the inputs to a layer.
- **MaxPooling**: Downsamples feature maps to reduce spatial dimensions, keeping the most important information.
- **Flatten Layer**: Converts the 2D feature maps into a 1D feature vector to feed into fully connected layers.
- **Dense Layer**: Fully connected layer that helps in decision making.
- **Dropout Layer**: Randomly disables some neurons during training to prevent overfitting.

---

### ⚙️ Methods Used in This Model

- **ImageDataGenerator**: Used for **data augmentation** — this generates new variations of existing images to improve the generalization of the model.
- **Model Compilation**:
    - **Optimizer**: `Adam` — efficient stochastic optimizer for training deep learning models.
    - **Loss Function**: `Binary Crossentropy` — suitable for binary classification problems.
- **Model Training**:
    - Using `.fit()` with data augmentation.
    - **Callback**: `ReduceLROnPlateau` lowers the learning rate when the validation loss stops improving.
- **Evaluation**:
    - Accuracy and loss on test set.
    - Confusion matrix for visual performance.
    - Classification report including precision, recall, and F1-score.
- **Prediction with Uploaded Image**:
    - Grayscale conversion.
    - Resizing to 150x150 pixels.
    - Normalization and reshaping.
    - Binary prediction (`NORMAL` or `PNEUMONIA`).

---

### 📂 Dataset Info

The dataset is sourced from the **Chest X-ray dataset** that contains two folders:
- `PNEUMONIA/` – Images diagnosed with pneumonia.
- `NORMAL/` – Healthy X-ray images.

---

### 📈 Model Goal

To assist medical professionals by providing a fast and reliable tool for detecting pneumonia from chest X-rays using AI.
""")


# --- Data Loading Function ---
def get_data(dir):
    data = []
    for label in labels:
        path = os.path.join(dir, label)
        if not os.path.exists(path):
            print(f"Warning: Directory not found - {path}")
            continue

        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                if img.lower().endswith(('jpeg', 'jpg', 'png')):
                    img_path = os.path.join(path, img)
                    img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img_arr is None:
                        print(f"Warning: Could not read image {img_path}")
                        continue
                    resize_arr = cv2.resize(img_arr, (img_resize, img_resize))
                    data.append([resize_arr, class_num])
            except Exception as e:
                print(f"Error processing {img}: {e}")
    return np.array(data, dtype='object')

# --- Load Data ---
base_dir = 'chest_xray'
if not os.path.exists(base_dir):
    raise FileNotFoundError(f"Base directory not found: {base_dir}")

train_path = os.path.join(base_dir, 'train')
test_path = os.path.join(base_dir, 'test')

train = get_data(train_path)
test = get_data(test_path)

# --- Data Preparation ---
x_train, y_train = [], []
x_test, y_test = [], []

for feat, label in train:
    x_train.append(feat)
    y_train.append(label)

for feat, label in test:
    x_test.append(feat)
    y_test.append(label)

# Normalize and reshape
x_train = np.array(x_train) / 255.0
x_test = np.array(x_test) / 255.0

x_train = x_train.reshape(-1, img_resize, img_resize, 1)
x_test = x_test.reshape(-1, img_resize, img_resize, 1)

y_train = np.array(y_train)
y_test = np.array(y_test)

# --- Image Data Augmentation ---
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False
)
datagen.fit(x_train)

# --- Build CNN Model ---
model = Sequential()

# 1st Convolution Block
model.add(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu', input_shape=(150, 150, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

# 2nd Convolution Block
model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

# 3rd Convolution Block
model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

# Flatten and Fully Connected Layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout helps reduce overfitting
model.add(Dense(1, activation='sigmoid'))  # Binary classification (PNEUMONIA vs NORMAL)

# --- Compile Model ---
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- Callbacks ---
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

# --- Train Model ---
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    validation_data=(x_test, y_test),
    epochs=5,
    callbacks=[reduce_lr]
)

# --- Evaluate Model ---
loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

# --- Classification Report ---
y_pred = (model.predict(x_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred, target_names=labels))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
st.pyplot(fig)

# --- Streamlit File Uploader ---
st.subheader("Try with your own X-ray image")
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption='Uploaded X-ray', use_column_width=True)
    
    # Button to trigger prediction
    if st.button('Predict'):
        # Preprocess the image
        img_array = np.array(image)
        img_array = cv2.resize(img_array, (img_resize, img_resize))
        img_array = img_array.reshape(1, img_resize, img_resize, 1) / 255.0
        
        # Make prediction
        prediction = model.predict(img_array)
        result = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
        confidence = prediction[0][0] if result == "PNEUMONIA" else 1 - prediction[0][0]
        
        st.subheader("Prediction Result")
        st.write(f"Result: {result}")
        st.write(f"Confidence: {confidence:.2%}")
