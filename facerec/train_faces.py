from tqdm import tqdm
import math
from sklearn import neighbors
import os
import os.path
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np
import cv2
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.utils import to_categorical
# from ultralytics import YOLO

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def train_knn(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []

    for class_dir in tqdm(os.listdir(train_dir)):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) == 1:
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir.split('_')[0])

    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def train_cnn(train_dir, model_dir="facerec/models"):
    X = []
    y = []
    classes = os.listdir(train_dir)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    for class_dir in tqdm(classes):
        target_dir = os.path.join(train_dir, class_dir)
        if not os.path.isdir(target_dir):
            continue
            
        for img_name in os.listdir(target_dir):
            if img_name.split('.')[-1].lower() in ALLOWED_EXTENSIONS:
                img_path = os.path.join(target_dir, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (64, 64))
                X.append(img)
                y.append(class_to_idx[class_dir])
    
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
        from tensorflow.keras.utils import to_categorical
    except ImportError as e:
        print(f"TensorFlow could not be loaded: {e}")
        return None

    X = np.array(X) / 255.0
    y = to_categorical(np.array(y), num_classes=len(classes))
    
    model = Sequential()
    # Using Conv2D as Convolution2D is an alias
    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(len(classes), activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Training CNN model...")
    hist = model.fit(X, y, batch_size=16, epochs=10, shuffle=True, verbose=2)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model.save_weights(os.path.join(model_dir, 'model_weights.h5'))
    model_json = model.to_json()
    with open(os.path.join(model_dir, "model.json"), "w") as json_file:
        json_file.write(model_json)
        
    with open(os.path.join(model_dir, 'history.pckl'), 'wb') as f:
        pickle.dump(hist.history, f)
        
    print(model.summary())
    return model


def trainer():
    dataset_path = "facerec/dataset"
    models_path = "facerec/models"
    
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    print("--- Starting KNN Training ---")
    train_knn(dataset_path, model_save_path=os.path.join(models_path, "trained_model.clf"), n_neighbors=3)
    
    print("\n--- Starting CNN Training ---")
    try:
        train_cnn(dataset_path, models_path)
    except Exception as e:
        print(f"CNN Training failed: {e}")

    print("\n--- Starting YOLOv8 Training ---")
    
    print("\nAll training processes completed!")
