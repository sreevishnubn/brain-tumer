# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings

# Define constants
DATA_DIR = 'D:/brain_tumor/Training/'
TEST_DIR_NO_TUMOR = 'D:/brain_tumor/Testing/no_tumor/'
TEST_DIR_PITUITARY_TUMOR = 'D:/brain_tumor/Testing/pituitary_tumor/'
CLASSES = {'no_tumor': 0, 'pituitary_tumor': 1}

# Helper function to load and preprocess images
def load_and_preprocess_images(directory):
    X = []
    Y = []
    for cls in CLASSES:
        pth = os.path.join(directory, cls)
        for j in os.listdir(pth):
            try:
                img = cv2.imread(os.path.join(pth, j), 0)  # Load image in grayscale
                img = cv2.resize(img, (200, 200))  # Resize image to 200x200 pixels
                X.append(img)
                Y.append(CLASSES[cls])
            except Exception as e:
                print(f"Error processing {os.path.join(pth, j)}: {str(e)}")
    X = np.array(X)
    Y = np.array(Y)
    X_updated = X.reshape(len(X), -1)  # Reshape image data for PCA
    return X_updated, Y

# Load and preprocess training data
xtrain, ytrain = load_and_preprocess_images(DATA_DIR)

# Split the data into train and test sets
xtrain, xtest, ytrain, ytest = train_test_split(xtrain, ytrain, random_state=10, test_size=0.20)

# Normalize pixel values using StandardScaler
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

# PCA Dimensionality Reduction
pca = PCA(n_components=0.98)  # Select the number of principal components to retain
xtrain_pca = pca.fit_transform(xtrain)
xtest_pca = pca.transform(xtest)

# Create and train the classifiers with hyperparameter tuning
param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10]}
param_grid_svm = {'C': [0.001, 0.01, 0.1, 1, 10], 'kernel': ['linear', 'rbf']}

# Perform GridSearchCV to find the best hyperparameters
grid_lr = GridSearchCV(LogisticRegression(), param_grid_lr, cv=5)
grid_svm = GridSearchCV(SVC(), param_grid_svm, cv=5)

# Suppress warnings during grid search
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    grid_lr.fit(xtrain_pca, ytrain)
    grid_svm.fit(xtrain_pca, ytrain)

# Get the best models
best_lr = grid_lr.best_estimator_
best_svm = grid_svm.best_estimator_

# Print the best hyperparameters
print("Best Logistic Regression Hyperparameters:", grid_lr.best_params_)
print("Best SVM Hyperparameters:", grid_svm.best_params_)

# Predictions
lr_pred = best_lr.predict(xtest_pca)
svm_pred = best_svm.predict(xtest_pca)

# Print accuracy scores and classification reports
print("Logistic Regression Testing Accuracy:", accuracy_score(ytest, lr_pred))
print("SVM Testing Accuracy:", accuracy_score(ytest, svm_pred))

print("Logistic Regression Classification Report:\n", classification_report(ytest, lr_pred))
print("SVM Classification Report:\n", classification_report(ytest, svm_pred))

# Visualization of test samples
dec = {0: 'No Tumor', 1: 'Positive Tumor'}
plt.figure(figsize=(12, 8))
c = 1

# Plot test samples with no tumor
for i in os.listdir(TEST_DIR_NO_TUMOR)[:9]:
    plt.subplot(3, 3, c)
    img = cv2.imread(os.path.join(TEST_DIR_NO_TUMOR, i), 0)  # Load test image in grayscale
    img1 = cv2.resize(img, (200, 200))  # Resize image to 200x200 pixels
    img1 = img1.reshape(1, -1)  # Reshape for PCA
    img1 = scaler.transform(img1)  # Normalize using the same scaler
    img1 = pca.transform(img1)  # Apply PCA
    p = best_svm.predict(img1)  # Predict using the SVM model
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c += 1

plt.figure(figsize=(12, 8))
c = 1

# Plot test samples with pituitary tumor
for i in os.listdir(TEST_DIR_PITUITARY_TUMOR)[:16]:
    plt.subplot(4, 4, c)
    img = cv2.imread(os.path.join(TEST_DIR_PITUITARY_TUMOR, i), 0)  # Load test image in grayscale
    img1 = cv2.resize(img, (200, 200))  # Resize image to 200x200 pixels
    img1 = img1.reshape(1, -1)  # Reshape for PCA
    img1 = scaler.transform(img1)  # Normalize using the same scaler
    img1 = pca.transform(img1)  # Apply PCA
    p = best_svm.predict(img1)  # Predict using the SVM model
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c += 1

# Show the plots
plt.show()
