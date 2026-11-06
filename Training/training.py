# ================================
# 🦴 Bone Fracture Detection Project (SVM-Centric Optimized)
# Models: SVM, Decision Tree, Random Forest
# Features: HOG + PCA
# Image Size: 64x64
# Dataset: Folder-based (fractured / not_fractured)
# ================================

import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================================
# Paths and save directory
# =========================================
base_path = "BoneFractureDataset/training"
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

# =========================================
# Function to load dataset (from folders)
# =========================================
def load_dataset_from_folders(base_dir, img_size=(64,64)):
    X = []
    y = []
    class_names = sorted(os.listdir(base_dir))
    
    for label_name in class_names:
        folder_path = os.path.join(base_dir, label_name)
        if not os.path.isdir(folder_path):
            continue
        
        print(f"\n📂 Loading images from: {label_name}")
        for img_name in tqdm(os.listdir(folder_path)):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            img = cv2.equalizeHist(cv2.resize(img, img_size))
            
            # Extract HOG features
            features = hog(img, orientations=9, pixels_per_cell=(8,8),
                           cells_per_block=(2,2), block_norm='L2-Hys', visualize=False)
            X.append(features)
            y.append(label_name)
    
    return np.array(X), np.array(y), class_names

# =========================================
# Load dataset
# =========================================
X, y, class_names = load_dataset_from_folders(base_path)
print("\n✅ Dataset loaded:")
print("Images:", X.shape, " Labels:", y.shape)
print("Classes:", class_names)

# =========================================
# Split Dataset (80% train, 20% test)
# =========================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)

# =========================================
# Preprocessing
# =========================================
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)
joblib.dump(encoder, os.path.join(save_dir, "label_encoder.pkl"))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA for dimensionality reduction
pca = PCA(n_components=150, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
joblib.dump(pca, os.path.join(save_dir, "pca_transformer.pkl"))

# =========================================
# Model evaluation function
# =========================================
def evaluate_model(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n📘 {model_name} Results:")
    print("Accuracy:", round(acc,4))
    
    print("Classification Report:\n",
          classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    return acc

# =========================================
# Train Models
# =========================================
print("\n🚀 Training Models...")

# 1️⃣ SVM
svm_model = SVC(kernel='linear', C=1, class_weight='balanced', probability=True, random_state=42)
svm_model.fit(X_train_pca, y_train_enc)
y_pred_svm = svm_model.predict(X_test_pca)
acc_svm = evaluate_model(y_test_enc, y_pred_svm, "SVM")
joblib.dump(svm_model, os.path.join(save_dir, "svm_model.pkl"))

# 2️⃣ Decision Tree
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train_pca, y_train_enc)
y_pred_dt = dt_model.predict(X_test_pca)
acc_dt = evaluate_model(y_test_enc, y_pred_dt, "Decision Tree")
joblib.dump(dt_model, os.path.join(save_dir, "decision_tree_model.pkl"))

# 3️⃣ Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_pca, y_train_enc)
y_pred_rf = rf_model.predict(X_test_pca)
acc_rf = evaluate_model(y_test_enc, y_pred_rf, "Random Forest")
joblib.dump(rf_model, os.path.join(save_dir, "random_forest_model.pkl"))

# 4 KNN
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_pca, y_train_enc)
y_pred_knn = knn_model.predict(X_test_pca)
acc_knn = evaluate_model(y_test_enc, y_pred_knn, "K Neighbors")
joblib.dump(knn_model, os.path.join(save_dir, "knn_model.pkl"))

# 5 Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train_pca, y_train_enc)
y_pred_logistic = logistic_model.predict(X_test_pca)
acc_logistic = evaluate_model(y_test_enc, y_pred_logistic, "Logistic Regression")
joblib.dump(logistic_model, os.path.join(save_dir, "logistic_model.pkl"))

# 6 svm+rf
svm_knn_model = StackingClassifier(
    estimators=[('svm', SVC(probability=True, kernel='rbf', C=1.0, random_state=42)),
                ('knn', KNeighborsClassifier(n_neighbors=3))],
    final_estimator=KNeighborsClassifier(n_neighbors=3),
    cv=5)

svm_knn_model.fit(X_train_pca, y_train_enc)
y_pred_svm_knn = svm_knn_model.predict(X_test_pca)
svm_rf_acc = evaluate_model(y_test_enc, y_pred_svm_knn, "SVM + KNN")
joblib.dump(svm_knn_model, os.path.join(save_dir, "svm_knn_model.pkl"))

# =========================================
# Compare Models
# =========================================
models = ['SVM', 'Decision Tree', 'Random Forest', 'K Neighbors','Logistic Regression', "SVM + KNN"]
accuracies = [acc_svm, acc_dt, acc_rf, acc_knn,acc_logistic,svm_rf_acc]

plt.figure(figsize=(6,4))
plt.bar(models, accuracies, color=['blue','green','orange', 'crimson','yellow','purple'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0,1)
for i, acc in enumerate(accuracies):
    plt.text(i, acc+0.01, f"{acc:.2f}", ha='center')
plt.show()

print("\n🏁 Training and Evaluation Complete!")
print("✅ Models, PCA, and LabelEncoder saved in:", save_dir)
