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
import time
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

def prepare_dataset(train_folders, output_folder, validation_split=0.2, task="detection"):
    """
    Prepare YOLO dataset with training and validation splits.
    
    Args:
        train_folders (dict): For detection/classification: {class_name: folder_path}
                            For segmentation: {class_name: (image_folder, mask_folder)}
        output_folder (str): Path to output dataset folder
        validation_split (float): Fraction of data to use for validation
        task (str): One of "detection", "classification", or "segmentation"
        
    Returns:
        tuple: (dataset_folder_path, class_mapping) or None if error occurs
    """
    output_folder = os.path.abspath(output_folder)
    
    # Clean up existing dataset folder if it exists
    if os.path.exists(output_folder):
        try:
            shutil.rmtree(output_folder)
            st.info(f"Cleaned up existing dataset folder: {output_folder}")
        except Exception as e:
            st.error(f"Error cleaning up dataset folder: {e}")
            return None, None
    
    try:
        os.makedirs(output_folder, exist_ok=True)
    except Exception as e:
        st.error(f"Error creating output folder: {e}")
        return None, None

    # Initialize class mapping
    class_mapping = {name: idx for idx, name in enumerate(train_folders.keys())}

    if task == "classification":
        # Classification specific directory structure
        train_path = os.path.join(output_folder, "train")
        val_path = os.path.join(output_folder, "val")
        
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)
        
        # Create class subdirectories and copy images
        for class_name, image_folder in train_folders.items():
            # Create class directories
            train_class_path = os.path.join(train_path, class_name)
            val_class_path = os.path.join(val_path, class_name)
            os.makedirs(train_class_path, exist_ok=True)
            os.makedirs(val_class_path, exist_ok=True)

            # Get all images
            image_files = [f for f in os.listdir(image_folder) 
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            
            if not image_files:
                st.warning(f"No images found in folder for class {class_name}")
                continue

            # Split into train and validation
            split_idx = int(len(image_files) * (1 - validation_split))
            train_images = image_files[:split_idx]
            val_images = image_files[split_idx:]

            # Copy training images
            for img in train_images:
                try:
                    shutil.copy2(
                        os.path.join(image_folder, img),
                        os.path.join(train_class_path, img)
                    )
                except Exception as e:
                    st.error(f"Error copying training image {img}: {e}")

            # Copy validation images
            for img in val_images:
                try:
                    shutil.copy2(
                        os.path.join(image_folder, img),
                        os.path.join(val_class_path, img)
                    )
                except Exception as e:
                    st.error(f"Error copying validation image {img}: {e}")

    elif task == "segmentation":
        # Segmentation specific directory structure
        train_images_path = os.path.join(output_folder, "train/images")
        train_masks_path = os.path.join(output_folder, "train/masks")
        train_labels_path = os.path.join(output_folder, "train/labels")
        val_images_path = os.path.join(output_folder, "val/images")
        val_masks_path = os.path.join(output_folder, "val/masks")
        val_labels_path = os.path.join(output_folder, "val/labels")
        
        # Create all necessary directories
        for path in [train_images_path, train_masks_path, train_labels_path,
                    val_images_path, val_masks_path, val_labels_path]:
            os.makedirs(path, exist_ok=True)

        for class_name, (image_folder, mask_folder) in train_folders.items():
            if not (os.path.exists(image_folder) and os.path.exists(mask_folder)):
                st.error(f"Source folders not found for class {class_name}")
                continue

            # Get all images and masks
            image_files = [f for f in os.listdir(image_folder) 
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            mask_files = [f for f in os.listdir(mask_folder) 
                         if f.lower().endswith((".png", ".jpg", ".jpeg"))]

            # Match images with masks
            valid_pairs = []
            for img_file in image_files:
                base_name = os.path.splitext(img_file)[0]
                matching_masks = [m for m in mask_files 
                                if os.path.splitext(m)[0] == base_name]
                if matching_masks:
                    valid_pairs.append((img_file, matching_masks[0]))
                else:
                    st.warning(f"No matching mask found for image {img_file}")

            if not valid_pairs:
                st.warning(f"No valid image-mask pairs found for class {class_name}")
                continue

            # Split into train and validation sets
            split_idx = int(len(valid_pairs) * (1 - validation_split))
            train_pairs = valid_pairs[:split_idx]
            val_pairs = valid_pairs[split_idx:]

            # Process training data
            for img_file, mask_file in train_pairs:
                try:
                    # Copy image
                    shutil.copy2(
                        os.path.join(image_folder, img_file),
                        os.path.join(train_images_path, img_file)
                    )
                    
                    # Copy mask
                    shutil.copy2(
                        os.path.join(mask_folder, mask_file),
                        os.path.join(train_masks_path, mask_file)
                    )
                    
                    # Create YOLO format label
                    label_file = f"{os.path.splitext(img_file)[0]}.txt"
                    with open(os.path.join(train_labels_path, label_file), "w") as f:
                        f.write(f"{class_mapping[class_name]} 0.5 0.5 1.0 1.0")
                except Exception as e:
                    st.error(f"Error processing training pair {img_file}: {e}")

            # Process validation data
            for img_file, mask_file in val_pairs:
                try:
                    # Copy image
                    shutil.copy2(
                        os.path.join(image_folder, img_file),
                        os.path.join(val_images_path, img_file)
                    )
                    
                    # Copy mask
                    shutil.copy2(
                        os.path.join(mask_folder, mask_file),
                        os.path.join(val_masks_path, mask_file)
                    )
                    
                    # Create YOLO format label
                    label_file = f"{os.path.splitext(img_file)[0]}.txt"
                    with open(os.path.join(val_labels_path, label_file), "w") as f:
                        f.write(f"{class_mapping[class_name]} 0.5 0.5 1.0 1.0")
                except Exception as e:
                    st.error(f"Error processing validation pair {img_file}: {e}")

    else:  # Detection task
        # Detection specific directory structure
        train_images_path = os.path.join(output_folder, "train/images")
        train_labels_path = os.path.join(output_folder, "train/labels")
        val_images_path = os.path.join(output_folder, "val/images")
        val_labels_path = os.path.join(output_folder, "val/labels")
        
        # Create directories
        for path in [train_images_path, train_labels_path, 
                    val_images_path, val_labels_path]:
            os.makedirs(path, exist_ok=True)

        for class_name, image_folder in train_folders.items():
            if not os.path.exists(image_folder):
                st.error(f"Source folder not found: {image_folder}")
                continue

            # Get all images
            image_files = [f for f in os.listdir(image_folder) 
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            
            if not image_files:
                st.warning(f"No images found in folder for class {class_name}")
                continue

            # Split into train and validation sets
            split_idx = int(len(image_files) * (1 - validation_split))
            train_images = image_files[:split_idx]
            val_images = image_files[split_idx:]

            # Process training images
            for img_file in train_images:
                try:
                    # Copy image
                    shutil.copy2(
                        os.path.join(image_folder, img_file),
                        os.path.join(train_images_path, img_file)
                    )
                    
                    # Create YOLO format label
                    label_file = f"{os.path.splitext(img_file)[0]}.txt"
                    with open(os.path.join(train_labels_path, label_file), "w") as f:
                        f.write(f"{class_mapping[class_name]} 0.5 0.5 1.0 1.0")
                except Exception as e:
                    st.error(f"Error processing training image {img_file}: {e}")

            # Process validation images
            for img_file in val_images:
                try:
                    # Copy image
                    shutil.copy2(
                        os.path.join(image_folder, img_file),
                        os.path.join(val_images_path, img_file)
                    )
                    
                    # Create YOLO format label
                    label_file = f"{os.path.splitext(img_file)[0]}.txt"
                    with open(os.path.join(val_labels_path, label_file), "w") as f:
                        f.write(f"{class_mapping[class_name]} 0.5 0.5 1.0 1.0")
                except Exception as e:
                    st.error(f"Error processing validation image {img_file}: {e}")

    st.success(f"Dataset prepared successfully in {output_folder}")
    return output_folder, class_mapping

def get_default_config():
    """Return default training configuration"""
    return {
        'epochs': 10,
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
                'epochs': st.number_input("Number of Epochs", 1, 1000, default_config['epochs']),
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

def get_yolo_models():
    """Return available YOLO model options with tasks"""
    return {
        "YOLOv11": {
            "detection": ["yolov11n", "yolov11s", "yolov11m", "yolov11l", "yolov11x"],
            "segmentation": ["yolov11n-seg", "yolov11s-seg", "yolov11m-seg", "yolov11l-seg", "yolov11x-seg"],
            "classification": ["yolov11n-cls", "yolov11s-cls", "yolov11m-cls", "yolov11l-cls", "yolov11x-cls"],
            "pose_estimation": ["yolov11n-pose", "yolov11s-pose", "yolov11m-pose", "yolov11l-pose", "yolov11x-pose"],
            "oriented_detection": ["yolov11n-obb", "yolov11s-obb", "yolov11m-obb", "yolov11l-obb", "yolov11x-obb"]
        },
        "YOLOv8": {
            "detection": ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
            "segmentation": ["yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg"],
            "classification": ["yolov8n-cls", "yolov8s-cls", "yolov8m-cls", "yolov8l-cls", "yolov8x-cls"]
        },
        "YOLOv7": {
            "detection": ["yolov7", "yolov7-tiny", "yolov7x"],
            "segmentation": ["yolov7-seg"]
        },
        "YOLOv6": {
            "detection": ["yolov6n", "yolov6s", "yolov6m", "yolov6l"]
        },
        "YOLOv5": {
            "detection": ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"],
            "segmentation": ["yolov5n-seg", "yolov5s-seg", "yolov5m-seg", "yolov5l-seg", "yolov5x-seg"]
        }
    }

def train_yolo(train_folders, epochs, model_type, use_gpu, save_model_path, training_config):
    """
    Train a YOLO model with specified parameters and progress tracking.
    
    Args:
        train_folders (dict): Training data folders
        epochs (int): Number of training epochs
        model_type (str): YOLO model type (e.g., yolov8n, yolov8n-cls, yolov8n-seg)
        use_gpu (bool): Whether to use GPU for training
        save_model_path (str): Path to save the trained model
        training_config (dict): Additional training configuration parameters
    
    Returns:
        YOLO: Trained model or None if training fails
    """
    try:
        # Create dataset folder
        dataset_folder = "prepared_dataset"
        dataset_folder = os.path.abspath(dataset_folder)

        # Progress bar for dataset preparation
        prep_progress = st.progress(0)
        st.write("Preparing dataset...")
        
        # Determine task type based on model name
        if "-cls" in model_type:
            task = "classification"
        elif "-seg" in model_type:
            task = "segmentation"
        else:
            task = "detection"
        
        # Prepare dataset with appropriate task type
        result = prepare_dataset(train_folders, dataset_folder, task=task)
        if result is None:
            st.error("Dataset preparation failed. Please check the errors above.")
            return None
            
        dataset_folder, class_mapping = result
        prep_progress.progress(100)

        # Initialize status display
        status_placeholder = st.empty()
        status_placeholder.write("Initializing training...")
        
        try:
            # Check GPU availability
            if use_gpu and not torch.cuda.is_available():
                st.warning("GPU was selected but is not available. Falling back to CPU.")
                use_gpu = False

            # Initialize model
            model = YOLO(model_type)
            
            # Create metrics display
            metrics_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            # Callback for updating training progress
            def update_progress(trainer):
                current_epoch = trainer.epoch + 1
                progress = int((current_epoch / epochs) * 100)
                progress_bar.progress(progress)
                
                metrics = trainer.metrics
                if task == "classification":
                    metrics_str = (
                        f"Epoch {current_epoch}/{epochs} | "
                        f"Accuracy: {metrics.get('metrics/accuracy', 0):.3f} | "
                        f"Loss: {metrics.get('train/loss', 0):.3f}"
                    )
                elif task == "segmentation":
                    metrics_str = (
                        f"Epoch {current_epoch}/{epochs} | "
                        f"mAP50: {metrics.get('metrics/mAP50(B)', 0):.3f} | "
                        f"Mask Loss: {metrics.get('train/mask_loss', 0):.3f} | "
                        f"Dice Score: {metrics.get('metrics/dice', 0):.3f}"
                    )
                else:  # detection
                    metrics_str = (
                        f"Epoch {current_epoch}/{epochs} | "
                        f"mAP50: {metrics.get('metrics/mAP50(B)', 0):.3f} | "
                        f"Box Loss: {metrics.get('train/box_loss', 0):.3f}"
                    )
                metrics_placeholder.write(metrics_str)

            # Base training arguments common to all tasks
            training_args = {
                "epochs": epochs,
                "imgsz": training_config['image_size'],
                "device": "0" if use_gpu else "cpu",
                "batch": training_config['batch_size'],
                "workers": training_config['num_workers'],
                "amp": training_config['use_amp'] and use_gpu,
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
                "resume": False,
                "model": model_type,
                "project": os.path.dirname(save_model_path),
                "name": os.path.basename(save_model_path).split('.')[0]
            }

            # Task-specific configurations
            if task == "classification":
                # For classification, directly use the dataset folder
                training_args["data"] = dataset_folder
            else:
                # Create YAML config for detection and segmentation
                data_yaml_path = os.path.join(dataset_folder, f"{model_type}_data.yaml")
                yaml_content = {
                    "path": dataset_folder,
                    "train": "train/images",
                    "val": "val/images",
                    "names": list(train_folders.keys())
                }

                if task == "segmentation":
                    yaml_content.update({
                        "train": {
                            "img": os.path.join(dataset_folder, "train/images"),
                            "mask": os.path.join(dataset_folder, "train/masks"),
                            "label": os.path.join(dataset_folder, "train/labels")
                        },
                        "val": {
                            "img": os.path.join(dataset_folder, "val/images"),
                            "mask": os.path.join(dataset_folder, "val/masks"),
                            "label": os.path.join(dataset_folder, "val/labels")
                        }
                    })
                    training_args.update({
                        "mask_ratio": 4,
                        "overlap_mask": True,
                        "mask_weights": 1.0
                    })

                # Save YAML configuration
                try:
                    import yaml
                    with open(data_yaml_path, "w") as f:
                        yaml.safe_dump(yaml_content, f, sort_keys=False, default_flow_style=False)
                    training_args["data"] = data_yaml_path
                except Exception as e:
                    st.error(f"Error creating YAML configuration: {e}")
                    return None

            # Register callback and start training
            model.add_callback('on_train_epoch_end', update_progress)

            try:
                # Start training
                status_placeholder.write("Training in progress...")
                model.train(**training_args)
                
                # Save the trained model
                model.save(save_model_path)
                status_placeholder.success("Training completed successfully!")
                
                return model

            except Exception as train_error:
                status_placeholder.error(f"Error during training: {train_error}")
                
                if use_gpu:
                    status_placeholder.warning("Attempting to fall back to CPU training...")
                    
                    # Modify configuration for CPU training
                    training_args.update({
                        "device": "cpu",
                        "batch": min(training_config['batch_size'], 4),  # Reduce batch size for CPU
                        "amp": False  # Disable AMP for CPU
                    })
                    
                    try:
                        # Retry training with CPU
                        model = YOLO(model_type)
                        model.add_callback('on_train_epoch_end', update_progress)
                        model.train(**training_args)
                        model.save(save_model_path)
                        status_placeholder.success("Training completed successfully on CPU!")
                        return model
                    except Exception as cpu_error:
                        status_placeholder.error(f"CPU training also failed: {cpu_error}")
                
                return None

        except Exception as model_error:
            status_placeholder.error(f"Error initializing model: {model_error}")
            return None

    except Exception as setup_error:
        st.error(f"Unexpected error during training setup: {setup_error}")
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
    """Test YOLO model with specified parameters, class filtering, GPU option, and FPS tracking"""
    model = YOLO(model_path)
    
    # If GPU is enabled, move the model to GPU
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    model.to(device)  # Move model to either GPU or CPU
    
    # Display the device being used
    st.write(f"Using device: {device.upper()}")  # Show whether GPU or CPU is being used

    # Create placeholders for FPS display and frame
    fps_placeholder = st.empty()
    frame_placeholder = st.empty()

    def process_frame(frame, fps_start_time):
        """Process a single frame with class filtering, GPU support, and FPS calculation"""
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

        # Calculate FPS
        fps = 1.0 / (time.time() - fps_start_time)
        
        # Draw FPS on frame
        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return annotated_frame, fps

    if input_source_type == "Webcam":
        # Add a button to start/stop the webcam
        if 'camera_on' not in st.session_state:
            st.session_state.camera_on = False
            
        if st.button("Start/Stop Camera"):
            st.session_state.camera_on = not st.session_state.camera_on
        
        # Only start the webcam if camera_on is True
        if st.session_state.camera_on:
            cap = cv2.VideoCapture(0)
            
            # Add FPS counter
            fps_list = []
            fps_update_interval = 0.5  # Update FPS display every 0.5 seconds
            last_fps_update = time.time()
            
            while st.session_state.camera_on:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access webcam!")
                    break

                # Process frame and get FPS
                start_time = time.time()
                annotated_frame, current_fps = process_frame(frame, start_time)
                fps_list.append(current_fps)
                
                # Update FPS display periodically
                if time.time() - last_fps_update > fps_update_interval:
                    avg_fps = sum(fps_list) / len(fps_list)
                    fps_placeholder.text(f"Average FPS: {avg_fps:.1f}")
                    fps_list = []
                    last_fps_update = time.time()

                # Display the frame
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            cap.release()
            frame_placeholder.empty()
            fps_placeholder.empty()
            st.info("Camera is off. Click 'Start/Stop Camera' to start.")
        else:
            st.info("Click 'Start/Stop Camera' to begin webcam detection")
    
    elif input_source_type == "Image":
        uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Detect"):
                start_time = time.time()
                annotated_frame, fps = process_frame(image, start_time)
                fps_placeholder.text(f"Processing Speed: {fps:.1f} FPS")
                frame_placeholder.image(annotated_frame, caption="Detection Result", use_container_width=True)
    
    elif input_source_type == "Video":
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
        if uploaded_file is not None:
            temp_file = "temp_video.mp4"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(temp_file)
            
            if 'video_playing' not in st.session_state:
                st.session_state.video_playing = True
                
            if st.button("Toggle Video"):
                st.session_state.video_playing = not st.session_state.video_playing

            # Add FPS counter
            fps_list = []
            fps_update_interval = 0.5  # Update FPS display every 0.5 seconds
            last_fps_update = time.time()

            while st.session_state.video_playing:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame and get FPS
                start_time = time.time()
                annotated_frame, current_fps = process_frame(frame, start_time)
                fps_list.append(current_fps)
                
                # Update FPS display periodically
                if time.time() - last_fps_update > fps_update_interval:
                    avg_fps = sum(fps_list) / len(fps_list)
                    fps_placeholder.text(f"Average FPS: {avg_fps:.1f}")
                    fps_list = []
                    last_fps_update = time.time()

                # Display the frame
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            cap.release()
            os.remove(temp_file)
            frame_placeholder.empty()
            fps_placeholder.empty()
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
        
        # Get available YOLO models
        yolo_models = get_yolo_models()
        
        # Model selection UI
        col1, col2 = st.columns(2)
        with col1:
            yolo_version = st.selectbox("YOLO Version", list(yolo_models.keys()))
        with col2:
            available_tasks = list(yolo_models[yolo_version].keys())
            task = st.selectbox("Task", available_tasks)
        
        # Model type selection based on version and task
        model_type = st.selectbox(
            "Model Type",
            yolo_models[yolo_version][task],
            help="Smaller models (n,s) are recommended for limited GPU memory"
        )
        
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

        use_gpu = st.checkbox("Use GPU for Training", value=torch.cuda.is_available())
        save_model_path = st.text_input("Save Model Path", "trained_model.pt")

        # Get training configuration
        training_config = get_training_config()
        
        # Display current configuration summary
        with st.expander("Current Training Configuration", expanded=False):
            st.json(training_config)

        if st.button("Start Training"):
            if not all(os.path.exists(folder) for folder in train_folders.values()):
                st.error("Some training folders are missing!")
            else:
                model = train_yolo(train_folders, training_config['epochs'], model_type, use_gpu, save_model_path, training_config)

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

        use_gpu = st.checkbox("Use GPU for Update", value=torch.cuda.is_available())
        save_model_path = st.text_input("Save Updated Model Path", "updated_model.pt")

        # Get training configuration
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
                update_model(existing_model_path, update_folders, training_config['epochs'], use_gpu, save_model_path, training_config)

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