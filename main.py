import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="👗 Fashion Recommender", page_icon="🛍️", layout="wide")

st.title("🧠 AI-Powered Fashion Recommender System")
st.markdown(
    """
    ### 👋 Upload an image to find visually similar fashion items!
    The system uses **ResNet50 (ImageNet pretrained)** to extract deep visual features
    and recommends **top 5 most similar items** from the fashion dataset.
    """
)

# ---------------------- LOAD FEATURES ----------------------
@st.cache_resource
def load_data():
    feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
    filenames = pickle.load(open('filenames.pkl', 'rb'))
    return feature_list, filenames

feature_list, filenames = load_data()

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])
    return model

model = load_model()

# ---------------------- UTILS ----------------------
def save_uploaded_file(uploaded_file):
    uploads_path = "uploads"
    os.makedirs(uploads_path, exist_ok=True)
    file_path = os.path.join(uploads_path, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices[0]

# ---------------------- FILE UPLOAD ----------------------
uploaded_file = st.file_uploader(
    "📤 Choose an image (jpg, jpeg, png, webp)...",
    type=['jpg', 'jpeg', 'png', 'webp']
)

if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    
    # Open uploaded image
    display_image = Image.open(uploaded_file)

    # Resize uploaded image to match recommended image size (e.g., 224x224)
    resized_display_image = display_image.resize((224, 224))

    # Display uploaded image
    st.image(resized_display_image, caption="Uploaded Image", width=224)

    # Feature extraction and recommendation
    with st.spinner("🔍 Extracting features and finding similar styles..."):
        features = feature_extraction(file_path, model)
        indices = recommend(features, feature_list)

    st.success("✅ Top 5 similar fashion items found!")

    # Display recommendations
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            rec_img = Image.open(filenames[indices[i]])
            rec_img_resized = rec_img.resize((224, 224))
            st.image(rec_img_resized, caption=f"Recommendation {i+1}", width=224)
else:
    st.info("👆 Upload an image above to get fashion recommendations.")
