# --- Silent installation of required packages ---
import os, sys, subprocess, logging, warnings, contextlib, importlib.util
import pandas as pd

print("⏳ Loading dependencies... Please wait.\n")

packages = [
    "tensorflow==2.15",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "numpy",
    "pandas"
]

for pkg in packages:
    try:
        # Check if the package is already available
        if importlib.util.find_spec(pkg.split("==")[0].replace("-", "_")) is None:
            raise ImportError
    except ImportError:
        try:
            # If not, install it silently
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--upgrade", pkg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"⚠️ Failed to install {pkg}. Please install it manually. Error: {e}")

# --- Suppress warnings & noisy logs for a clean output ---
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.FATAL)
warnings.filterwarnings("ignore")

# ===========================
# GPU CHECK & SETUP
# ===========================
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Use only the first GPU and allow memory growth to prevent crashes
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ GPU is available and enabled: {gpus[0].name}")
    except RuntimeError as e:
        print("⚠️ GPU setup failed, running on CPU:", e)
else:
    print("\n❌ No GPU detected, running on CPU.")

# ===========================
# Standard Imports
# ===========================
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ===========================
# CONFIGURATION
# ===========================
BASE_DIR = "chest_xray/chest_xray" # Adjust if your folder structure is different
train_dir = os.path.join(BASE_DIR, "train")
val_dir = os.path.join(BASE_DIR, "val")
test_dir = os.path.join(BASE_DIR, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10 # Increased epochs for better training, EarlyStopping will prevent overfitting
PLOTS_DIR = "plots" # Directory to save all generated plots and reports

# Create plots directory if it doesn't exist
os.makedirs(PLOTS_DIR, exist_ok=True)


# ===========================
# DATA GENERATORS
# ===========================
def create_generators(batch_size, target_size):
    """Creates data generators with augmentation for training and without for validation/testing."""
    # Generator with data augmentation for the training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    # Generator with only rescaling for validation and test sets
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=target_size, batch_size=batch_size, class_mode='binary'
    )
    val_generator = val_test_datagen.flow_from_directory(
        val_dir, target_size=target_size, batch_size=batch_size, class_mode='binary'
    )
    test_generator = val_test_datagen.flow_from_directory(
        test_dir, target_size=target_size, batch_size=batch_size, class_mode='binary', shuffle=False
    )
    print("\n✅ Data generators created.")
    return train_generator, val_generator, test_generator


# ===========================
# MODEL CREATION
# ===========================
def build_custom_cnn(input_shape):
    """Builds the custom CNN model from the first script."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid') # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("✅ Custom CNN model built.")
    return model

def build_densenet(input_shape, fine_tune=False):
    """Builds a DenseNet121 model for feature extraction or fine-tuning."""
    base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=output)

    if not fine_tune:
        # Freeze all layers of the base model
        for layer in base_model.layers:
            layer.trainable = False
        print("✅ DenseNet121 (Base) model built for feature extraction.")
    else:
        # Fine-tune: Unfreeze the top layers of the base model
        for layer in base_model.layers[-50:]:
            layer.trainable = True
        print("✅ DenseNet121 (Fine-Tuned) model built.")

    model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model


# ===========================
# CUSTOM CALLBACK
# ===========================
class NewLineLogger(tf.keras.callbacks.Callback):
    """A custom callback to print a new line at the end of each epoch for cleaner logs."""
    def on_epoch_end(self, epoch, logs=None):
        print()


# ===========================
# ROBUST TRAINING FUNCTION
# ===========================
def safe_training(model, train_gen, val_gen, epochs, model_name):
    """Handles training with callbacks and saves the best model."""
    print(f"\n🚀 Starting training for {model_name}...")
    
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, verbose=1, restore_best_weights=True),
        ModelCheckpoint(f"{model_name}.h5", monitor="val_loss", save_best_only=True, verbose=1),
        NewLineLogger()  # Add the custom callback for clean logging
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )
    print(f"\n✅ Training for {model_name} complete.")
    return model, history


# ===========================
# VISUALIZATION & REPORTING
# ===========================
def plot_training_history(history, model_name):
    """Plots and saves the training & validation accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'{model_name} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Plot Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'{model_name} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    filename = os.path.join(PLOTS_DIR, f"{model_name}_history.png")
    plt.savefig(filename)
    print(f"📁 Saved training history plot to: {filename}")
    plt.show(block=False)
    plt.pause(3)
    plt.close()

def plot_dashboard(model, model_name, test_gen):
    """Generates, saves, and displays a confusion matrix and classification report."""
    print(f"\n📊 Generating dashboard for {model_name}...")
    y_true = test_gen.classes
    y_pred_proba = model.predict(test_gen).ravel()
    y_pred = (y_pred_proba > 0.5).astype("int32")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=test_gen.class_indices.keys(),
                yticklabels=test_gen.class_indices.keys())
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    cm_filename = os.path.join(PLOTS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_filename)
    print(f"📁 Saved confusion matrix to: {cm_filename}")
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    # Classification Report
    print(f"\nClassification Report - {model_name}:\n")
    class_names = list(test_gen.class_indices.keys())
    report_str = classification_report(y_true, y_pred, target_names=class_names)
    print(report_str)
    
    # Save report to CSV
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    report_filename = os.path.join(PLOTS_DIR, f"{model_name}_classification_report.csv")
    df_report.to_csv(report_filename)
    print(f"📊 Saved classification report to: {report_filename}")

    # Calculate AUC
    auc = roc_auc_score(y_true, y_pred_proba)
    print(f"Area Under Curve (AUC): {auc:.4f}")
    return auc


# ===========================
# MAIN EXECUTION
# ===========================
if __name__ == "__main__":
    train_gen, val_gen, test_gen = create_generators(BATCH_SIZE, IMG_SIZE)
    input_shape = (*IMG_SIZE, 3)

    models_to_run = {
        "Custom_CNN": build_custom_cnn(input_shape),
        "DenseNet121_Base": build_densenet(input_shape, fine_tune=False),
        "DenseNet121_FineTuned": build_densenet(input_shape, fine_tune=True)
    }

    trained_models = {}
    histories = {}
    
    choice = input("\n👉 Do you want to train models from scratch? (y/n): ").strip().lower()

    for name, model in models_to_run.items():
        model_path = f"{name}.h5"
        if choice == 'n' and os.path.exists(model_path):
            print(f"\n✅ Found saved model '{model_path}'. Loading it...")
            trained_models[name] = load_model(model_path)
            histories[name] = None  # No history available when loading
        else:
            if choice == 'y':
                print(f"\n⚡ Training {name} from scratch as requested.")
            else:
                print(f"\n⚠️ Saved model '{model_path}' not found. Training a new one.")
                
            trained_model, history = safe_training(model, train_gen, val_gen, EPOCHS, name)
            trained_models[name] = trained_model
            histories[name] = history

    # --- Evaluation and Comparison ---
    print("\n\n" + "="*50)
    print("PERFORMANCE EVALUATION & COMPARISON")
    print("="*50)

    results = []
    for name, model in trained_models.items():
        print(f"\n--- Evaluating {name} on the test set ---")
        loss, accuracy = model.evaluate(test_gen, verbose=1)
        auc = plot_dashboard(model, name, test_gen)
        results.append({
            "Model": name,
            "Test Accuracy": accuracy,
            "Test Loss": loss,
            "AUC": auc
        })
        
        # Plot training history if the model was trained in this session
        if histories[name]:
            plot_training_history(histories[name], name)

    # --- Final Summary Table ---
    print("\n\n" + "="*50)
    print("🏆 FINAL MODEL COMPARISON SUMMARY 🏆")
    print("="*50)
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    summary_filename = os.path.join(PLOTS_DIR, "model_comparison_summary.csv")
    df_results.to_csv(summary_filename, index=False)
    print(f"\n\n📁 Final comparison summary saved to: {summary_filename}")