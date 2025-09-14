import streamlit as st
import os
import shutil
import cv2
import numpy as np
from PIL import Image
import tempfile
import torch
from pathlib import Path
import time
import yaml
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1
import onnxruntime
import json

def get_yolo_models():
    """Return available YOLO model options focused on detection"""
    return {
        "YOLOv11": {
            "detection": ["yolov11n", "yolov11s", "yolov11m", "yolov11l", "yolov11x"]
        },
        "YOLOv10": {
            "detection": ["yolov10n", "yolov10s", "yolov10m", "yolov10l", "yolov10x"]
        },
        "YOLOv9": {
            "detection": ["yolov9n", "yolov9s", "yolov9m", "yolov9l", "yolov9x"]
        },
        "YOLOv8": {
            "detection": ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
        },
        "YOLOv7": {
            "detection": ["yolov7", "yolov7-tiny", "yolov7x"]
        },
        "YOLOv6": {
            "detection": ["yolov6n", "yolov6s", "yolov6m", "yolov6l"]
        },
        "YOLOv5": {
            "detection": ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"]
        }
    }

def get_default_config():
    """Return default training configuration"""
    return {
        'epochs': 100,
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

def handle_model_upload(uploaded_file):
    """Handle uploaded model file and display its information"""
    if uploaded_file is not None:
        # Create a temporary directory to save the uploaded model
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            model_path = tmp_file.name
            
        try:
            if model_path.endswith('.pt'):
                # Load YOLO model
                model = YOLO(model_path)
                classes = model.names
                
                # Display classes
                st.success("YOLO model loaded successfully!")
                st.write("Model Classes:")
                st.write(f"Total number of classes: {len(classes)}")
                
                # Display classes in a grid
                num_cols = 3
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
                
            elif model_path.endswith('.onnx'):
                # Load ONNX model
                session = onnxruntime.InferenceSession(model_path)
                inputs = session.get_inputs()
                outputs = session.get_outputs()
                
                # Display ONNX model info
                st.success("ONNX model loaded successfully!")
                st.write("Model Information:")
                st.write(f"Input name: {inputs[0].name}")
                st.write(f"Input shape: {inputs[0].shape}")
                st.write(f"Output name: {outputs[0].name}")
                
                # Try to load class names if available
                class_file = Path(model_path).with_suffix('.json')
                if class_file.exists():
                    with open(class_file, 'r') as f:
                        classes = json.load(f)
                else:
                    classes = None
                    
                return model_path, classes
                
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None
    return None, None

def save_model_to_onnx(model, save_path, input_shape=(640, 640)):
    """Convert and save model to ONNX format"""
    try:
        # Export to ONNX with dynamic batch size
        model.export(format='onnx', dynamic=True, imgsz=input_shape)
        
        # Save class names separately
        class_file = Path(save_path).with_suffix('.json')
        with open(class_file, 'w') as f:
            json.dump(model.names, f)
            
        st.success(f"Model successfully converted to ONNX and saved at {save_path}")
        st.info(f"Class names saved at {class_file}")
    except Exception as e:
        st.error(f"Error converting model to ONNX: {e}")

class FaceRecognitionTrainer:
    def __init__(self):
        self.mtcnn = MTCNN(keep_all=True)
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
        
    def prepare_face_dataset(self, train_folders):
        """Prepare face recognition dataset using MTCNN for face detection"""
        processed_faces = {}
        progress_bar = st.progress(0)
        total_folders = len(train_folders)
        
        for idx, (person_name, folder_path) in enumerate(train_folders.items()):
            faces = []
            image_files = list(Path(folder_path).glob('*.jpg'))
            st.write(f"Processing images for {person_name}...")
            
            for img_path in image_files:
                img = Image.open(img_path)
                face = self.mtcnn(img)
                if face is not None:
                    faces.append(face)
                    
            if faces:
                processed_faces[person_name] = faces
            else:
                st.warning(f"No faces detected in images for {person_name}")
                
            progress_bar.progress((idx + 1) / total_folders)
            
        return processed_faces
        
    def train(self, processed_faces, save_path):
        """Train face recognition model and save embeddings"""
        embeddings = {}
        progress_bar = st.progress(0)
        total_persons = len(processed_faces)
        
        for idx, (person_name, faces) in enumerate(processed_faces.items()):
            st.write(f"Computing embeddings for {person_name}...")
            person_embeddings = []
            
            for face in faces:
                with torch.no_grad():
                    embedding = self.facenet(face.unsqueeze(0))
                    person_embeddings.append(embedding)
                    
            embeddings[person_name] = torch.stack(person_embeddings).mean(0)
            progress_bar.progress((idx + 1) / total_persons)
        
        # Save the model and embeddings
        torch.save({
            'mtcnn_state_dict': self.mtcnn.state_dict(),
            'facenet_state_dict': self.facenet.state_dict(),
            'embeddings': embeddings
        }, save_path)
        
        return embeddings

def test_models(model_paths, input_source_type, confidence_threshold, selected_classes, use_gpu):
    """Test both YOLO, ONNX, and face recognition models"""
    models = []
    for path in model_paths:
        if path.endswith('.pt'):
            if 'face_recognition' in path.lower():
                # Load face recognition model
                checkpoint = torch.load(path)
                mtcnn = MTCNN(keep_all=True)
                mtcnn.load_state_dict(checkpoint['mtcnn_state_dict'])
                facenet = InceptionResnetV1(pretrained='vggface2').eval()
                facenet.load_state_dict(checkpoint['facenet_state_dict'])
                embeddings = checkpoint['embeddings']
                if use_gpu and torch.cuda.is_available():
                    mtcnn = mtcnn.to('cuda')
                    facenet = facenet.to('cuda')
                models.append(('face', mtcnn, facenet, embeddings))
            else:
                # Load YOLO model
                yolo_model = YOLO(path)
                if use_gpu and torch.cuda.is_available():
                    yolo_model.to('cuda')
                models.append(('yolo', yolo_model))
        elif path.endswith('.onnx'):
            # Load ONNX model
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(path, providers=providers)
            
            # Try to load class names
            class_file = Path(path).with_suffix('.json')
            if class_file.exists():
                with open(class_file, 'r') as f:
                    classes = json.load(f)
            else:
                classes = None
            models.append(('onnx', session, classes))

    fps_placeholder = st.empty()
    frame_placeholder = st.empty()

    def process_frame(frame, fps_start_time):
        """Process frame with all loaded models"""
        results_frame = frame.copy()
        
        for model_type, *model_parts in models:
            if model_type == 'yolo':
                yolo_model = model_parts[0]
                results = yolo_model.predict(
                    frame,
                    conf=confidence_threshold,
                    classes=selected_classes,
                    device='cuda' if use_gpu else 'cpu'
                )
                results_frame = results[0].plot()
                
            elif model_type == 'face':
                mtcnn, facenet, stored_embeddings = model_parts
                boxes, _ = mtcnn.detect(frame)
                if boxes is not None:
                    for box in boxes:
                        face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                        face_tensor = mtcnn(Image.fromarray(face))
                        if face_tensor is not None:
                            if use_gpu:
                                face_tensor = face_tensor.cuda()
                            embedding = facenet(face_tensor.unsqueeze(0))
                            
                            # Find closest match
                            min_dist = float('inf')
                            best_match = None
                            for name, stored_emb in stored_embeddings.items():
                                if use_gpu:
                                    stored_emb = stored_emb.cuda()
                                dist = torch.dist(embedding, stored_emb)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_match = name
                            
                            # Draw box and label
                            cv2.rectangle(results_frame, 
                                        (int(box[0]), int(box[1])), 
                                        (int(box[2]), int(box[3])), 
                                        (0, 255, 0), 2)
                            cv2.putText(results_frame, 
                                      f"{best_match} ({min_dist:.2f})", 
                                      (int(box[0]), int(box[1])-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                      (0, 255, 0), 2)
                                      
            elif model_type == 'onnx':
                session, classes = model_parts
                # Preprocess image
                input_name = session.get_inputs()[0].name
                input_shape = session.get_inputs()[0].shape
                img = cv2.resize(frame, (input_shape[2], input_shape[3]))
                img = img.transpose((2, 0, 1))
                img = np.expand_dims(img, 0)
                img = img.astype(np.float32) / 255.0
                
                # Run inference
                outputs = session.run(None, {input_name: img})
                
                # Post-process outputs
                boxes = outputs[0][0]
                for box in boxes:
                    score = box[4]
                    if score > confidence_threshold:
                        x1, y1, x2, y2 = box[0:4]
                        cls_id = int(box[5])
                        label = classes[cls_id] if classes else str(cls_id)
                        
                        cv2.rectangle(results_frame,
                                    (int(x1), int(y1)),
                                    (int(x2), int(y2)),
                                    (255, 0, 0), 2)
                        cv2.putText(results_frame,
                                  f"{label} ({score:.2f})",
                                  (int(x1), int(y1)-10),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.9, (255, 0, 0), 2)

        fps = 1.0 / (time.time() - fps_start_time)
        cv2.putText(results_frame, f"FPS: {fps:.1f}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return results_frame, fps

    if input_source_type == "Webcam":
        if 'camera_on' not in st.session_state:
            st.session_state.camera_on = False
            
        if st.button("Start/Stop Camera"):
            st.session_state.camera_on = not st.session_state.camera_on
            
        if st.session_state.camera_on:
            cap = cv2.VideoCapture(0)
            fps_list = []
            fps_update_interval = 0.5
            last_fps_update = time.time()
            
            while st.session_state.camera_on:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access webcam!")
                    break

                start_time = time.time()
                annotated_frame, current_fps = process_frame(frame, start_time)
                fps_list.append(current_fps)
                
                if time.time() - last_fps_update > fps_update_interval:
                    avg_fps = sum(fps_list) / len(fps_list)
                fps_placeholder.text(f"Average FPS: {avg_fps:.1f}")
                fps_list.clear()
                last_fps_update = time.time()
                
                frame_placeholder.image(annotated_frame, channels="BGR")
                
            cap.release()
            
    elif input_source_type == "Video":
        uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            fps_list = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                start_time = time.time()
                annotated_frame, current_fps = process_frame(frame, start_time)
                fps_list.append(current_fps)
                
                frame_placeholder.image(annotated_frame, channels="BGR")
                fps_placeholder.text(f"Current FPS: {current_fps:.1f}")
                
            cap.release()
            os.unlink(tfile.name)
            
    elif input_source_type == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            start_time = time.time()
            annotated_frame, fps = process_frame(frame, start_time)
            frame_placeholder.image(annotated_frame, channels="BGR")

def main():
    st.title("Multi-Model Testing Interface")
    
    # Sidebar for model selection and settings
    st.sidebar.header("Settings")
    
    # Model upload section
    st.sidebar.subheader("Model Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload model files (.pt or .onnx)",
        type=['pt', 'onnx'],
        accept_multiple_files=True
    )
    
    model_paths = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            model_path, classes = handle_model_upload(uploaded_file)
            if model_path:
                model_paths.append(model_path)
    
    # Face Recognition Training Section
    st.sidebar.subheader("Face Recognition Training")
    train_face = st.sidebar.checkbox("Enable Face Recognition Training")
    
    if train_face:
        st.sidebar.markdown("### Training Settings")
        dataset_path = st.sidebar.text_input("Dataset Path (one folder per person)")
        
        if st.sidebar.button("Start Training"):
            if dataset_path and os.path.exists(dataset_path):
                # Create face recognition trainer
                trainer = FaceRecognitionTrainer()
                
                # Get person folders
                person_folders = {
                    folder.name: str(folder)
                    for folder in Path(dataset_path).iterdir()
                    if folder.is_dir()
                }
                
                if person_folders:
                    # Prepare dataset
                    processed_faces = trainer.prepare_face_dataset(person_folders)
                    
                    if processed_faces:
                        # Train model
                        save_path = os.path.join(dataset_path, "face_recognition_model.pt")
                        embeddings = trainer.train(processed_faces, save_path)
                        
                        # Convert to ONNX if requested
                        if st.sidebar.checkbox("Convert to ONNX"):
                            onnx_path = os.path.join(dataset_path, "face_recognition_model.onnx")
                            save_model_to_onnx(trainer.facenet, onnx_path)
                            
                        st.success("Face recognition model training completed!")
                else:
                    st.error("No person folders found in dataset path")
            else:
                st.error("Invalid dataset path")
    
    # Testing settings
    st.sidebar.subheader("Testing Settings")
    input_source_type = st.sidebar.selectbox(
        "Select Input Source",
        ["Webcam", "Video", "Image"]
    )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        0.0, 1.0, 0.5, 0.05
    )
    
    use_gpu = st.sidebar.checkbox(
        "Use GPU (if available)",
        value=torch.cuda.is_available()
    )
    
    selected_classes = None
    if model_paths:
        # Get available classes from the first YOLO model
        for path in model_paths:
            if path.endswith('.pt') and 'face_recognition' not in path.lower():
                model = YOLO(path)
                class_list = list(model.names.values())
                selected_classes = st.sidebar.multiselect(
                    "Select classes to detect",
                    options=range(len(class_list)),
                    format_func=lambda x: class_list[x]
                )
                break
    
    # Start testing
    if st.sidebar.button("Start Testing") and model_paths:
        test_models(
            model_paths,
            input_source_type,
            confidence_threshold,
            selected_classes,
            use_gpu
        )

if __name__ == "__main__":
    main()