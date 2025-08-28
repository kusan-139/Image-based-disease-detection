# app.py
import os
import sys
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report

# --- Suppress TensorFlow logging ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ===========================
# PATHS CONFIGURATION
# ===========================
# Determine the base directory, works for both script and PyInstaller executable
BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))

# Define paths to data and models
test_dir = os.path.join(BASE_DIR, "chest_xray", "chest_xray", "test")
model_cnn_path = os.path.join(BASE_DIR, "Custom_CNN.h5")
model_base_path = os.path.join(BASE_DIR, "DenseNet121_Base.h5")
model_ft_path = os.path.join(BASE_DIR, "DenseNet121_FineTuned.h5")

# Model and Image settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_NAMES = ["Custom CNN", "DenseNet121 Base", "DenseNet121 Fine-Tuned"]

# ===========================
# DATA PIPELINE
# ===========================
# Use st.cache_data to avoid reloading the generator on every interaction
@st.cache_resource
def get_test_generator():
    # ... function code
    if not os.path.exists(test_dir):
        st.error(f"Test data directory not found at: {test_dir}")
        return None
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )
    return test_generator

test_generator = get_test_generator()

# ===========================
# LOAD MODELS
# ===========================
# Use st.cache_resource for loading models as it's a non-serializable object
@st.cache_resource
def load_all_models():
    models = {}
    model_paths = {
        MODEL_NAMES[0]: model_cnn_path,
        MODEL_NAMES[1]: model_base_path,
        MODEL_NAMES[2]: model_ft_path
    }
    all_models_found = True
    for name, path in model_paths.items():
        if not os.path.exists(path):
            st.warning(f"⚠️ Model not found: {os.path.basename(path)}. Please train it first.")
            all_models_found = False
        else:
            models[name] = load_model(path)

    if all_models_found:
        st.success("✅ All three pre-trained models loaded successfully!")
    return models if all_models_found else None

models = load_all_models()

# ===========================
# EVALUATION FUNCTION
# ===========================
# Use st.cache_data to cache evaluation results
@st.cache_data
def evaluate_model(model_name):
    # This function is now designed to be cached.
    # It fetches the model from the global `models` dictionary.
    model = models[model_name]
    loss, acc = model.evaluate(test_generator, verbose=0)
    y_true = test_generator.classes
    y_pred_proba = model.predict(test_generator).ravel()
    y_pred = (y_pred_proba > 0.5).astype("int32")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=test_generator.class_indices.keys(),
        output_dict=True
    )
    return acc, cm, report

# ===========================
# STREAMLIT DASHBOARD
# ===========================

st.title("🩺 X-Ray Disease Detection Model Comparison")
st.markdown("Comparison of **Custom CNN vs. DenseNet Base vs. Fine-tuned** models.")

# --- Main conditional check: Only show the app UI if models and data are loaded ---
if models and test_generator:
    
    # Button to trigger evaluation
    if st.button("⚡ Run Full Model Evaluation"):
        evaluation_results = {}
        with st.spinner("Evaluating all models... This may take a moment. ⏳"):
            for name in MODEL_NAMES:
                evaluation_results[name] = evaluate_model(name)

        # Accuracy Comparison
        st.subheader("📶 Accuracy Comparison")
        accuracies = [evaluation_results[name][0] for name in MODEL_NAMES]
        fig_acc = px.bar(
            x=MODEL_NAMES,
            y=accuracies,
            labels={"x": "Model", "y": "Accuracy"},
            color=MODEL_NAMES,
            text=[f"{acc:.4f}" for acc in accuracies]
        )
        st.plotly_chart(fig_acc)

        # Confusion Matrices
        st.subheader("📌 Confusion Matrices")
        for name in MODEL_NAMES:
            cm = evaluation_results[name][1]
            fig_cm = ff.create_annotated_heatmap(
                z=cm,
                x=list(test_generator.class_indices.keys()),
                y=list(test_generator.class_indices.keys()),
                colorscale="Blues",
                showscale=True
            )
            fig_cm.update_layout(title=f"Confusion Matrix - {name}")
            st.plotly_chart(fig_cm)

        # Classification Reports
        st.subheader("📑 Classification Reports")
        for name in MODEL_NAMES:
            with st.expander(f"View Report for {name}"):
                st.json(evaluation_results[name][2])
    else:
        st.info("👆 Click **Run Full Model Evaluation** to compare all models on the test set.")

    # ==========================
    # Single Image Prediction 
    # ==========================
    st.write("---") 

    col1, col2 = st.columns([1, 8])
    with col1:
        st.image("x-ray.png", width=100) # Ensure 'x-ray.png' is in the correct folder
    with col2:
        st.markdown("<h1 style='text-align: left;'>Test a Single Image </h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        try:
            img = load_img(uploaded_file, target_size=IMG_SIZE)
            if img.mode != "RGB":
                img = img.convert("RGB")

            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            st.write("")
            st.write("#### Predictions:")
            
            class_names = list(test_generator.class_indices.keys())

            for name, model in models.items():
                prob = model.predict(img_array, verbose=0)[0][0]
                pred_class_idx = int(prob > 0.5)
                pred_class_name = class_names[pred_class_idx]
                st.write(f"**{name}:** `{pred_class_name}` (Confidence: {prob:.4f})")

        except Exception as e:
            st.error(f"⚠️ Could not process the image. Please try another one. Error: {e}")

# --- These messages will ONLY show if the 'if' condition above is False ---
elif not test_generator:
    st.error("Could not load the test data. Please check the directory path.")
else:
    st.warning("Please run the training script to generate all three `.h5` model files.")
