import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import re

# Load your trained model
model = tf.keras.models.load_model('Leaves.h5')  # Update with your actual model path

# Class names
labels = [
    'Alstonia Scholaris (P2)', 'Apple', 'Apta', 'Arjun (P1)','Chinar (P11)', 'Eggplant',
    'Gauva (P3)', 'Groundnut', 'Indian Rubber Tree', 'Jamun (P5)', 'Jatropha (P6)','karanj', 'Kashid',
    'Lemon', 'Mango', 'Nilgiri', 'Pimpal', 'Pomegranate', 'Pongamia Pinnata (P7)', 'Sita Ashok',
    'Sonmohar', 'Vad', 'Vilayati Chinch', 'Beans', 'Sugarcane'
]

def predict(image):
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return labels[np.argmax(score)], 100 * np.max(score)

def normalize_text(text):
    # Convert to lowercase, replace underscores and hyphens with spaces, and remove non-alphanumeric characters
    return re.sub(r'\W+', ' ', text.lower().replace('_', ' ').replace('-', ' '))

st.title("Leaf Classification App")
st.write("Upload an image of a leaf to classify")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Extract actual label from file name if available
    actual_label = None
    file_name = os.path.basename(uploaded_file.name)
    normalized_file_name = normalize_text(file_name)

    for label in labels:
        normalized_label = normalize_text(label)
        if normalized_label in normalized_file_name:
            actual_label = label
            break

    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', width=300)
    st.write("")

    # Preprocess and predict
    image = image.resize((299, 299))
    label, confidence = predict(image)

    st.write("### Classification Results")
    if actual_label:
        st.write(f"**Actual Label:** {actual_label}")
    st.write(f"**Predicted Label:** {label}")
    
    
    # Add some visual flair
   
    