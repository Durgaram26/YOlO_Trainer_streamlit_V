import streamlit as st
import os
from ultralytics import YOLO
import shutil
import cv2
import numpy as np
from PIL import Image
import tempfile
import torch
from pathlib import Path

def handle_model_upload(uploaded_file):
    """Handle uploaded model file and display its classes in a grid layout"""
    if uploaded_file is not None:
        # Create a temporary directory to save the uploaded model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            model_path = tmp_file.name
            
        try:
            # Load model and get classes
            model = YOLO(model_path)
            classes = model.names
            
            # Display classes in a grid layout
            st.success("Model loaded successfully!")
            st.write("Model Classes:")
            
            # Display total number of classes
            st.write(f"Total number of classes: {len(classes)}")
            
            # Calculate number of columns (we'll show 3 classes per row)
            num_cols = 3
            
            # Create rows of classes
            for i in range(0, len(classes), num_cols):
                cols = st.columns(num_cols)
                for j in range(num_cols):
                    idx = i + j
                    if idx < len(classes):
                        with cols[j]:
                            st.markdown(f"""
                            <div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; text-align: center;'>
                                <b>Class {idx}</b><br>
                                {classes[idx]}
                            </div>
                            """, unsafe_allow_html=True)
            
            return model_path, classes
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None
    return None, None

def prepare_dataset(image_folders, output_folder, validation_split=0.2):
    """Prepare YOLO dataset with training and validation splits."""
    # Ensure the output folder path is absolute and correctly defined
    output_folder = os.path.abspath(output_folder)  # Make sure it's an absolute path
    os.makedirs(output_folder, exist_ok=True)

    # Define subdirectories for train and validation images and labels
    train_images_path = os.path.join(output_folder, "train/images")
    train_labels_path = os.path.join(output_folder, "train/labels")
    val_images_path = os.path.join(output_folder, "val/images")
    val_labels_path = os.path.join(output_folder, "val/labels")

    # Create subdirectories if they don't exist
    os.makedirs(train_images_path, exist_ok=True)
    os.makedirs(train_labels_path, exist_ok=True)
    os.makedirs(val_images_path, exist_ok=True)
    os.makedirs(val_labels_path, exist_ok=True)

    class_mapping = {}
    for class_id, (class_name, folder) in enumerate(image_folders.items()):
        class_mapping[class_name] = class_id
        image_files = [f for f in os.listdir(folder) if f.endswith((".jpg", ".jpeg", ".png"))]

        # Split into training and validation sets
        split_index = int(len(image_files) * (1 - validation_split))
        training_images = image_files[:split_index]
        validation_images = image_files[split_index:]

        # Process training images
        for image_file in training_images:
            shutil.copy(os.path.join(folder, image_file), train_images_path)
            label_file = f"{os.path.splitext(image_file)[0]}.txt"
            label_path = os.path.join(train_labels_path, label_file)

            with open(label_path, "w") as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0")

        # Process validation images
        for image_file in validation_images:
            shutil.copy(os.path.join(folder, image_file), val_images_path)
            label_file = f"{os.path.splitext(image_file)[0]}.txt"
            label_path = os.path.join(val_labels_path, label_file)

            with open(label_path, "w") as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0")

    return output_folder, class_mapping

def get_default_config():
    """Return default training configuration"""
    return {
        'image_size': 640,
        'batch_size': 16,
        'num_workers': 8,
        'learning_rate': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'patience': 50,
        'save_period': -1,
        'use_amp': True,
        'use_cache': False,
        'use_pretrained': True,
        'optimizer': "SGD",
        'seed': 42
    }

def get_training_config():
    """Get training configuration with default values if not explicitly set"""
    default_config = get_default_config()
    
    # Create an expander for advanced configuration
    with st.expander("Advanced Training Configuration", expanded=False):
        st.info("Configure advanced training parameters. If not modified, default values will be used.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            config = {
                'image_size': st.number_input("Image Size", 320, 1280, default_config['image_size'], 32),
                'batch_size': st.number_input("Batch Size", 1, 64, default_config['batch_size']),
                'num_workers': st.number_input("Number of Workers", 1, 16, default_config['num_workers']),
                'learning_rate': st.number_input("Learning Rate", 0.0001, 0.1, default_config['learning_rate'], format="%.4f"),
                'momentum': st.number_input("Momentum", 0.0, 1.0, default_config['momentum']),
                'weight_decay': st.number_input("Weight Decay", 0.0, 0.001, default_config['weight_decay'], format="%.4f"),
            }
        
        with col2:
            config.update({
                'warmup_epochs': st.number_input("Warmup Epochs", 0, 10, default_config['warmup_epochs']),
                'patience': st.number_input("Early Stopping Patience", 0, 100, default_config['patience']),
                'save_period': st.number_input("Save Period (epochs)", -1, 100, default_config['save_period']),
                'use_amp': st.checkbox("Use Automatic Mixed Precision", default_config['use_amp']),
                'use_cache': st.checkbox("Use Cache", default_config['use_cache']),
                'use_pretrained': st.checkbox("Use Pretrained Weights", default_config['use_pretrained']),
                'optimizer': st.selectbox("Optimizer", ["SGD", "Adam", "AdamW"], index=0),
                'seed': st.number_input("Random Seed", 0, 1000, default_config['seed'])
            })
        
        st.session_state.config_modified = True
        return config
    
    # If the expander wasn't opened, return default config
    if not st.session_state.get('config_modified', False):
        return default_config
    return st.session_state.get('current_config', default_config)

def train_yolo(train_folders, epochs, model_type, use_gpu, save_model_path, training_config):
    """Train a YOLO model with specified parameters and progress tracking."""
    dataset_folder = "prepared_dataset"
    dataset_folder = os.path.abspath(dataset_folder)

    # Progress bar for dataset preparation
    prep_progress = st.progress(0)
    st.write("Preparing dataset...")
    
    prepare_dataset(train_folders, dataset_folder)
    prep_progress.progress(100)

    data_yaml_path = os.path.join(dataset_folder, f"{model_type}_data.yaml")
    with open(data_yaml_path, "w") as f:
        f.write(f"""
train: {os.path.join(dataset_folder, 'train/images')}
val: {os.path.join(dataset_folder, 'val/images')}
nc: {len(train_folders)}
names: {list(train_folders.keys())}
""")

    status_placeholder = st.empty()
    status_placeholder.write("Initializing training...")
    
    try:
        model = YOLO(model_type)
        
        # Create metrics display
        metrics_placeholder = st.empty()
        progress_bar = st.progress(0)

        # Combine base training args with custom config
        training_args = {
            "data": data_yaml_path,
            "epochs": epochs,
            "imgsz": training_config['image_size'],
            "device": "0" if use_gpu else "cpu",
            "workers": training_config['num_workers'],
            "batch": training_config['batch_size'],
            "amp": training_config['use_amp'],
            "patience": training_config['patience'],
            "save": True,
            "save_period": training_config['save_period'],
            "cache": training_config['use_cache'],
            "exist_ok": True,
            "pretrained": training_config['use_pretrained'],
            "optimizer": training_config['optimizer'],
            "lr0": training_config['learning_rate'],
            "momentum": training_config['momentum'],
            "weight_decay": training_config['weight_decay'],
            "warmup_epochs": training_config['warmup_epochs'],
            "verbose": True,
            "seed": training_config['seed'],
            "deterministic": True
        }

        # Update progress using Streamlit's progress tracking
        def update_progress(trainer):
            current_epoch = trainer.epoch + 1
            progress = int((current_epoch / epochs) * 100)
            progress_bar.progress(progress)
            
            metrics = trainer.metrics
            metrics_str = (
                f"Epoch {current_epoch}/{epochs} | "
                f"mAP50: {metrics.get('metrics/mAP50(B)', 0):.3f} | "
                f"Loss: {metrics.get('train/box_loss', 0):.3f}"
            )
            metrics_placeholder.write(metrics_str)

        # Train the model
        model.add_callback('on_train_epoch_end', update_progress)
        model.train(**training_args)
        model.save(save_model_path)
        
        status_placeholder.success("Training completed successfully!")
        return model

    except Exception as e:
        status_placeholder.error(f"Error during training: {e}")
        if use_gpu:
            status_placeholder.warning("Attempting to fall back to CPU training...")
            training_args["device"] = "cpu"
            training_args["batch"] = min(training_config['batch_size'], 4)
            training_args["amp"] = False
            
            try:
                model = YOLO(model_type)
                model.add_callback('on_train_epoch_end', update_progress)
                model.train(**training_args)
                model.save(save_model_path)
                status_placeholder.success("Training completed successfully on CPU!")
                return model
            except Exception as cpu_e:
                status_placeholder.error(f"CPU training also failed: {cpu_e}")
        
        return None

def update_model(existing_model_path, train_folders, epochs, use_gpu, save_model_path, training_config):
    """Update an existing YOLO model with new training data and progress tracking."""
    try:
        status_placeholder = st.empty()
        status_placeholder.write("Loading existing model...")
        
        model = YOLO(existing_model_path)
        existing_classes = model.names
        
        # Prepare dataset with progress tracking
        prep_progress = st.progress(0)
        status_placeholder.write("Preparing dataset...")
        
        dataset_folder = "prepared_dataset"
        dataset_folder = os.path.abspath(dataset_folder)
        prepare_dataset(train_folders, dataset_folder)
        prep_progress.progress(100)

        # Combine classes and create YAML
        new_classes = list(train_folders.keys())
        all_classes = list(set(existing_classes + new_classes))
        
        data_yaml_path = os.path.join(dataset_folder, "update_data.yaml")
        with open(data_yaml_path, "w") as f:
            f.write(f"""
train: {os.path.join(dataset_folder, 'train/images')}
val: {os.path.join(dataset_folder, 'val/images')}
nc: {len(all_classes)}
names: {all_classes}
""")

        # Setup training arguments
        training_args = {
            "data": data_yaml_path,
            "epochs": epochs,
            "imgsz": training_config['image_size'],
            "device": "0" if use_gpu else "cpu",
            "workers": training_config['num_workers'],
            "batch": training_config['batch_size'],
            "amp": training_config['use_amp'],
            "patience": training_config['patience'],
            "save": True,
            "save_period": training_config['save_period'],
            "cache": training_config['use_cache'],
            "exist_ok": True,
            "optimizer": training_config['optimizer'],
            "lr0": training_config['learning_rate'],
            "momentum": training_config['momentum'],
            "weight_decay": training_config['weight_decay'],
            "warmup_epochs": training_config['warmup_epochs'],
            "verbose": True,
            "seed": training_config['seed']
        }

        # Add progress tracking
        metrics_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        def on_train_epoch_end(trainer):
            current_epoch = trainer.epoch + 1
            progress = int((current_epoch / epochs) * 100)
            progress_bar.progress(progress)
            
            metrics = trainer.metrics
            metrics_str = (
                f"Epoch {current_epoch}/{epochs} | "
                f"mAP50: {metrics.get('metrics/mAP50(B)', 0):.3f} | "
                f"Loss: {metrics.get('train/box_loss', 0):.3f}"
            )
            metrics_placeholder.write(metrics_str)
            
        training_args['callbacks'] = {'on_train_epoch_end': on_train_epoch_end}

        # Train the model
        status_placeholder.write("Updating model...")
        model.train(**training_args)
        
        # Save the updated model
        model.save(save_model_path)
        status_placeholder.success("Model updated successfully!")
        return model
        
    except Exception as e:
        status_placeholder.error(f"Error during model update: {e}")
        return None

def test_model(model_path, input_source_type, confidence_threshold, selected_classes, use_gpu):
    """Test YOLO model with specified parameters, class filtering, and GPU option"""
    model = YOLO(model_path)
    
    # If GPU is enabled, move the model to GPU
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    model.to(device)  # Move model to either GPU or CPU
    
    # Display the device being used
    st.write(f"Using device: {device.upper()}")  # Show whether GPU or CPU is being used

    def process_frame(frame):
        """Process a single frame with class filtering and GPU support"""
        if selected_classes == "all" or selected_classes == ["all"]:
            classes_to_detect = None  # None means detect all classes
        else:
            classes_to_detect = selected_classes

        # Use the device set above (GPU if enabled, otherwise CPU)
        results = model.predict(
            frame,
            conf=confidence_threshold,
            classes=classes_to_detect,
            device=device
        )

        return results[0].plot()

    if input_source_type == "Webcam":
        # Add a button to start/stop the webcam
        if 'camera_on' not in st.session_state:
            st.session_state.camera_on = False
            
        if st.button("Start/Stop Camera"):
            st.session_state.camera_on = not st.session_state.camera_on
        
        # Only start the webcam if camera_on is True
        if st.session_state.camera_on:
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            
            while st.session_state.camera_on:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access webcam!")
                    break

                annotated_frame = process_frame(frame)
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(rgb_frame, channels="RGB", use_container_width=True)

            cap.release()
            stframe.empty()
            st.info("Camera is off. Click 'Start/Stop Camera' to start.")
        else:
            st.info("Click 'Start/Stop Camera' to begin webcam detection")
    
    elif input_source_type == "Image":
        uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Detect"):
                annotated_image = process_frame(image)
                st.image(annotated_image, caption="Detection Result", use_container_width=True)
    
    elif input_source_type == "Video":
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
        if uploaded_file is not None:
            temp_file = "temp_video.mp4"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(temp_file)
            stframe = st.empty()
            
            if 'video_playing' not in st.session_state:
                st.session_state.video_playing = True
                
            if st.button("Toggle Video"):
                st.session_state.video_playing = not st.session_state.video_playing

            while st.session_state.video_playing:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame = process_frame(frame)
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(rgb_frame, channels="RGB", use_container_width=True)

            cap.release()
            os.remove(temp_file)
            stframe.empty()
            st.info("Video playback stopped.")


def main():
    st.title("YOLO Image Trainer, Tester, and Updater")
    
    # Initialize session state for config
    if 'config_modified' not in st.session_state:
        st.session_state.config_modified = False
    if 'current_config' not in st.session_state:
        st.session_state.current_config = get_default_config()
    
    option = st.selectbox("Choose an option", ["Train Model", "Update Existing Model", "Test Model"])

    if option == "Train Model":
        st.subheader("Train New Model")
        
        # Training configuration
        train_folders = {}
        num_train_classes = st.number_input("Number of Classes for Training", min_value=1, step=1, value=1)
        
        for i in range(num_train_classes):
            col1, col2 = st.columns(2)
            with col1:
                class_name = st.text_input(f"Class {i + 1} Name", key=f"train_class_{i}")
            with col2:
                folder_path = st.text_input(f"Folder Path for Class {i + 1}", key=f"train_folder_{i}")
            if class_name and folder_path:
                train_folders[class_name] = folder_path

        if torch.cuda.is_available():
            st.info(f"GPU available with {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB memory")

        model_type = st.selectbox(
            "Model Type",
            ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
            help="Smaller models (n,s) are recommended for limited GPU memory"
        )
        
        epochs = st.slider("Number of Epochs", 1, 100, 10)
        use_gpu = st.checkbox("Use GPU for Training", value=torch.cuda.is_available())
        save_model_path = st.text_input("Save Model Path", "trained_model.pt")

        # Get training configuration (will use defaults if advanced config not opened)
        training_config = get_training_config()
        
        # Display current configuration summary
        with st.expander("Current Training Configuration", expanded=False):
            st.json(training_config)

        if st.button("Start Training"):
            if not all(os.path.exists(folder) for folder in train_folders.values()):
                st.error("Some training folders are missing!")
            else:
                model = train_yolo(train_folders, epochs, model_type, use_gpu, save_model_path, training_config)

    elif option == "Update Existing Model":
        st.subheader("Update Existing Model")
        
        uploaded_model = st.file_uploader(
            "Drop your existing model here (.pt file)",
            type=['pt'],
            help="Drag and drop your existing YOLO model file"
        )
        
        existing_model_path = None
        if uploaded_model:
            existing_model_path, existing_classes = handle_model_upload(uploaded_model)

        update_folders = {}
        num_update_classes = st.number_input("Number of New Classes to Add", min_value=1, step=1, value=1)
        
        for i in range(num_update_classes):
            col1, col2 = st.columns(2)
            with col1:
                class_name = st.text_input(f"New Class {i + 1} Name", key=f"update_class_{i}")
            with col2:
                folder_path = st.text_input(f"Folder Path for Class {i + 1}", key=f"update_folder_{i}")
            if class_name and folder_path:
                update_folders[class_name] = folder_path

        epochs = st.slider("Number of Epochs for Update", 1, 100, 10)
        use_gpu = st.checkbox("Use GPU for Update", value=torch.cuda.is_available())
        save_model_path = st.text_input("Save Updated Model Path", "updated_model.pt")

        # Get training configuration (will use defaults if advanced config not opened)
        training_config = get_training_config()
        
        # Display current configuration summary
        with st.expander("Current Training Configuration", expanded=False):
            st.json(training_config)

        if st.button("Start Model Update"):
            if not all(os.path.exists(folder) for folder in update_folders.values()):
                st.error("Some update folders are missing!")
            elif existing_model_path is None:
                st.error("Please upload an existing model first!")
            else:
                update_model(existing_model_path, update_folders, epochs, use_gpu, save_model_path, training_config)

    elif option == "Test Model":
        st.subheader("Test Model")
        
        uploaded_test_model = st.file_uploader(
            "Drop your model here (.pt file)",
            type=['pt'],
            help="Drag and drop your YOLO model file for testing"
        )
        
        model_path = None
        model_classes = None
        if uploaded_test_model:
            model_path, model_classes = handle_model_upload(uploaded_test_model)

        if model_path:
            # Test configuration
            input_source_type = st.radio("Select Input Source", ["Webcam", "Image", "Video"])
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
            
            # GPU Enable checkbox
            use_gpu = st.checkbox("Use GPU for Inference", value=torch.cuda.is_available())
            
            if model_classes:
                class_options = ["all"] + list(range(len(model_classes)))
                selected_classes = st.multiselect(
                    "Select Classes to Detect",
                    options=class_options,
                    default="all",
                    format_func=lambda x: f"Class {x}: {model_classes[x]}" if x != "all" else "All Classes"
                )
            else:
                selected_classes = "all"

            test_model(model_path, input_source_type, confidence_threshold, selected_classes, use_gpu)
        else:
            st.warning("Please upload a model first!")  

if __name__ == "__main__":
    main()