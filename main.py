import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import joblib

# === Data Preparation ===
train_directory = r'C:\Users\Teja Sri\OneDrive\Desktop\mp\brain_tumor_dataset\training'
validation_directory = r'C:\Users\Teja Sri\OneDrive\Desktop\mp\brain_tumor_dataset\validation'
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_directory, target_size=(150, 150), batch_size=32, class_mode='binary'
)
validation_generator = test_datagen.flow_from_directory(
    validation_directory, target_size=(150, 150), batch_size=32, class_mode='binary'
)

# === CNN Model for Feature Extraction ===
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)
cnn_model.save('cnn_model.h5')

# === Feature Extraction Function ===
def extract_features(generator):
    features, labels = [], []
    for imgs, labs in generator:
        feats = cnn_model.predict(imgs)
        features.append(feats)
        labels.append(labs)
        if len(features)*generator.batch_size >= generator.samples:
            break
    return np.vstack(features), np.hstack(labels)

X_train, y_train = extract_features(train_generator)
X_val,   y_val   = extract_features(validation_generator)
X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat   = X_val.reshape(len(X_val),   -1)

# === Train Traditional ML Models ===
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train_flat, y_train)
svm = SVC(kernel='linear').fit(X_train_flat, y_train)
rf  = RandomForestClassifier(n_estimators=100).fit(X_train_flat, y_train)

ensemble = VotingClassifier(
    estimators=[('knn', knn), ('svm', svm), ('rf', rf)],
    voting='hard'
).fit(X_train_flat, y_train)

# === Predictions ===
knn_pred      = knn.predict(X_val_flat)
svm_pred      = svm.predict(X_val_flat)
rf_pred       = rf.predict(X_val_flat)
ensemble_pred = ensemble.predict(X_val_flat)

# === Accuracy Scores ===
print(f"KNN Accuracy:           {accuracy_score(y_val, knn_pred)*100:.2f}%")
print(f"SVM Accuracy:           {accuracy_score(y_val, svm_pred)*100:.2f}%")
print(f"Random Forest Accuracy: {accuracy_score(y_val, rf_pred)*100:.2f}%")
print(f"Ensemble Accuracy:      {accuracy_score(y_val, ensemble_pred)*100:.2f}%")

# === Confusion Matrices & Classification Reports ===
models = {
    "KNN":      (knn_pred,),
    "SVM":      (svm_pred,),
    "RandomForest": (rf_pred,),
    "Ensemble": (ensemble_pred,)
}

for name, (preds,) in models.items():
    cm = confusion_matrix(y_val, preds)
    cr = classification_report(y_val, preds, target_names=['No Tumor','Tumor'])
    print(f"\n{name} Confusion Matrix:\n{cm}")
    print(f"\n{name} Classification Report:\n{cr}")

# === Save Models ===
joblib.dump(knn, "knn_model.pkl")
joblib.dump(svm, "svm_model.pkl")
joblib.dump(rf,  "rf_model.pkl")
joblib.dump(ensemble, "ensemble_model.pkl")
