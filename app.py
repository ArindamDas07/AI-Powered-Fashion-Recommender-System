import os
import pickle
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
import tensorflow as tf

# -----------------------------
# ✅ GPU Configuration
# -----------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU detected: {gpus[0].name}")
    try:
        for gpu in gpus:
            # Allow TensorFlow to grow GPU memory as needed
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("🚀 GPU memory growth enabled")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ No GPU detected. Running on CPU.")

# -----------------------------
# ✅ Load pretrained ResNet50 model (feature extractor)
# -----------------------------
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = Sequential([base_model, GlobalMaxPooling2D()])

# -----------------------------
# ✅ Feature extraction function
# -----------------------------
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = np.expand_dims(image.img_to_array(img), axis=0)
    img_array = preprocess_input(img_array)
    # Predict on GPU
    feature_vector = model.predict(img_array, verbose=0).flatten()
    return feature_vector / norm(feature_vector)

# -----------------------------
# ✅ Process all images
# -----------------------------
image_dir = "images"
filenames = [
    os.path.join(image_dir, file)
    for file in os.listdir(image_dir)
    if file.lower().endswith(('.jpg', '.jpeg', '.png'))
]

print(f"📸 Found {len(filenames)} images for feature extraction")

feature_list = []
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    for file in tqdm(filenames, desc="Extracting features on GPU"):
        feature_list.append(extract_features(file, model))

# -----------------------------
# ✅ Save results
# -----------------------------
with open("embeddings.pkl", "wb") as f_embed, open("filenames.pkl", "wb") as f_names:
    pickle.dump(feature_list, f_embed)
    pickle.dump(filenames, f_names)

print(f"✅ Feature extraction complete for {len(filenames)} images.")
print("💾 Saved: embeddings.pkl and filenames.pkl")
