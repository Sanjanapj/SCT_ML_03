
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Path to the extracted Kaggle dataset
data_dir = "path_to_kaggle_dataset/train"  # <-- Replace this with your local path

# Parameters
img_size = 64  # Resize images to 64x64
X = []
y = []

# Load and preprocess images
for img_file in os.listdir(data_dir):
    if img_file.endswith(".jpg"):
        label = 1 if "dog" in img_file else 0
        img_path = os.path.join(data_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_size, img_size))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X.append(img_gray.flatten())  # Convert 2D image to 1D
        y.append(label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM classifier
print("Training the SVM model...")
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Predict on test data
y_pred = svm_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))
