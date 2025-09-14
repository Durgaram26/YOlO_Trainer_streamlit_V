import streamlit as st
import os
from ultralytics import YOLO
import shutil
import cv2
import numpy as np
from PIL import Image
import tempfile
import torch
# Function to handle uploaded model file and display classes
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

# Function to prepare dataset
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

def train_yolo(train_folders, epochs, model_type, use_gpu, save_model_path):
    """Train a YOLO model with specified parameters and CUDA fixes."""
    dataset_folder = "prepared_dataset"
    dataset_folder = os.path.abspath(dataset_folder)  # Convert to absolute path

    # Prepare the dataset
    prepare_dataset(train_folders, dataset_folder)

    # Generate the YAML file for training
    data_yaml_path = os.path.join(dataset_folder, f"{model_type}_data.yaml")
    with open(data_yaml_path, "w") as f:
        f.write(f"""
train: {os.path.join(dataset_folder, 'train/images')}
val: {os.path.join(dataset_folder, 'val/images')}
nc: {len(train_folders)}
names: {list(train_folders.keys())}
""")

    print("Training YOLO model...")

    try:
        # Load the YOLO model
        model = YOLO(model_type)

        # Configure training parameters with CUDA fixes
        training_args = {
            "data": data_yaml_path,
            "epochs": epochs,
            "imgsz": 640,
            "device": "0" if use_gpu else "cpu",
            "workers": 8,
            "batch": 16,  # Reduced batch size to prevent CUDA memory issues
            "amp": False,  # Disable automatic mixed precision to avoid CUDA issues
            "patience": 50,
            "save": True,
            "save_period": -1,
            "cache": False,
            "exist_ok": True,
            "pretrained": True,
            "optimizer": "SGD",  # Use SGD instead of default optimizer
            "verbose": True,
            "seed": 0,
            "deterministic": True
        }

        # Add CUDA-specific configurations if GPU is enabled
        if use_gpu:
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = True
                training_args.update({
                    "device": "0",
                    "batch": 8,  # Further reduce batch size for CUDA
                })
            else:
                raise RuntimeError("CUDA is not available. Check your GPU or drivers.")

        # Train the model with the configured parameters
        model.train(**training_args)

        # Save the trained model
        model.save(save_model_path)
        print(f"Model trained successfully and saved to: {save_model_path}")
        return model

    except RuntimeError as e:
        print(f"RuntimeError during training: {e}")
        
        # Attempt to train with CPU if GPU fails
        if use_gpu:
            print("Attempting to fall back to CPU training...")
            training_args["device"] = "cpu"
            training_args["batch"] = 8
            training_args["amp"] = False
            
            try:
                model = YOLO(model_type)
                model.train(**training_args)
                model.save(save_model_path)
                print(f"Model trained successfully on CPU and saved to: {save_model_path}")
                return model
            except Exception as cpu_e:
                print(f"CPU training also failed: {cpu_e}")
                
        return None

def update_model(existing_model_path, train_folders, epochs, use_gpu, save_model_path):
    """Update an existing YOLO model with new training data."""
    try:
        # Load the existing YOLO model
        model = YOLO(existing_model_path)
        print(f"Loaded existing model: {existing_model_path}")

        # Get existing class names
        existing_classes = model.names
        print("Existing model classes:", existing_classes)

        # Prepare the dataset
        dataset_folder = "prepared_dataset"
        dataset_folder = os.path.abspath(dataset_folder)  # Convert to absolute path

        # Prepare the dataset (this ensures paths are correct and dataset is split correctly)
        prepare_dataset(train_folders, dataset_folder)

        # Combine existing and new classes
        new_classes = list(train_folders.keys())
        all_classes = list(set(existing_classes + new_classes))

        # Generate the YAML file
        data_yaml_path = os.path.join(dataset_folder, "update_data.yaml")
        with open(data_yaml_path, "w") as f:
            f.write(f"""
train: {os.path.join(dataset_folder, 'train/images')}
val: {os.path.join(dataset_folder, 'val/images')}
nc: {len(all_classes)}
names: {all_classes}
""")

        print("Updating YOLO model...")
        device = "0" if use_gpu else "cpu"

        # Train the model with new data
        model.train(data=data_yaml_path, epochs=epochs, imgsz=640, device=device)
        print("Model updated successfully!")

        # Save the updated model
        model.save(save_model_path)
        print(f"Updated model saved to: {save_model_path}")
        return model
    except Exception as e:
        print(f"Error during model update: {e}")
        return None

# Function to test model
def test_model(model_path, input_source_type, confidence_threshold, selected_classes):
    """Test YOLO model with specified parameters and class filtering"""
    model = YOLO(model_path)
    
    def process_frame(frame):
        """Process a single frame with class filtering"""
        # Handle "all" classes case correctly
        if selected_classes == "all" or selected_classes == ["all"]:
            classes_to_detect = None  # None means detect all classes
        else:
            classes_to_detect = selected_classes
        
        # Predict with class filtering
        results = model.predict(
            frame,
            conf=confidence_threshold,
            classes=classes_to_detect  # Will be None for all classes
        )
        
        return results[0].plot()

    if input_source_type == "Webcam":
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        
        if 'camera_on' not in st.session_state:
            st.session_state.camera_on = True
            
        if st.button("Toggle Camera"):
            st.session_state.camera_on = not st.session_state.camera_on
            
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
        st.info("Camera is off. Click 'Toggle Camera' to start.")
    
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

# Main Streamlit app
def main():
    st.title("YOLO Image Trainer, Tester, and Updater")
    
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

        # Add CUDA memory usage warning
        if torch.cuda.is_available():
            st.warning("CUDA is available. Training will start with optimized settings to prevent memory issues.")
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            st.info(f"Available GPU memory: {gpu_mem:.2f} GB")

        model_type = st.selectbox(
            "Model Type",
            ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
            help="Smaller models (n,s) are recommended for limited GPU memory"
        )
        
        epochs = st.slider("Number of Epochs", 1, 100, 10)
        use_gpu = st.checkbox("Use GPU for Training", value=torch.cuda.is_available())
        save_model_path = st.text_input("Save Model Path", "trained_model.pt")

        if st.button("Start Training"):
            if not all(os.path.exists(folder) for folder in train_folders.values()):
                st.error("Some training folders are missing!")
            else:
                with st.spinner("Training in progress..."):
                    model = train_yolo(train_folders, epochs, model_type, use_gpu, save_model_path)
                    if model is not None:
                        st.success("Training completed successfully!")
                    else:
                        st.error("Training failed. Please check the logs for details.")

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
        use_gpu = st.checkbox("Use GPU for Update", value=True)
        save_model_path = st.text_input("Save Updated Model Path", "updated_model.pt")

        if st.button("Start Model Update"):
            if not all(os.path.exists(folder) for folder in update_folders.values()):
                st.error("Some update folders are missing!")
            elif existing_model_path is None:
                st.error("Please upload an existing model first!")
            else:
                update_model(existing_model_path, update_folders, epochs, use_gpu, save_model_path)

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

            test_model(model_path, input_source_type, confidence_threshold, selected_classes)
        else:
            st.warning("Please upload a model first!")

if __name__ == "__main__":
    main()