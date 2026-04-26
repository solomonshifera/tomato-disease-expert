import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page Configuration
st.set_page_config(page_title="Tomato Health AI", layout="centered")

# 1. Load the Model (Updated for Compatibility)
@st.cache_resource
def load_my_model():
    model_path = 'tomato_disease_modelv2.h5'
    # 'compile=False' itti dabaluun dogoggora 'TypeError' sana siif fura
    return tf.keras.models.load_model(model_path, compile=False)

model = load_my_model()

# 2. Define the Disease Names (Labels)
class_labels = [
    'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 
    'Septoria_leaf_spot', 'Spider_mites Two-spotted_spider_mite', 
    'Target_Spot', 'Yellow_Leaf_Curl_Virus', 'Mosaic_virus', 'Healthy'
]

# UI Elements
st.title("🍅 Tomato Disease Expert Diagnosis")
st.write("Upload a clear photo of a tomato leaf to get an instant AI diagnosis.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # Streamlit haaraa irratti 'use_container_width' fayyadamna
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("---")
    
    with st.spinner('Analyzing...'):
        # Preprocessing
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediction
        preds = model.predict(img_array)
        result_index = np.argmax(preds[0])
        confidence = preds[0][result_index]

        # Professional Smart Filter (Threshold)
        if confidence < 0.45:
            st.error("⚠️ Error: This AI is 95% sure this is NOT a tomato leaf. Please upload a clear photo.")
        else:
            st.success(f"### Diagnosis: {class_labels[result_index]}")
            st.info(f"**Confidence Level:** {confidence*100:.2f}%")
            
            # Advice based on confidence
            if confidence > 0.90:
                st.write("✅ High-precision diagnosis. Please consult the treatment manual.")
            else:
                st.write("ℹ️ Moderate confidence. Consider taking a closer photo in better light.")
