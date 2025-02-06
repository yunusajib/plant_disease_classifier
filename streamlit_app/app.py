# streamlit_app/app.py
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import yaml
from pathlib import Path
import sys
import os
import time
from datetime import datetime
import pandas as pd

# Add project root to path and get absolute paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = ROOT_DIR / 'configs' / 'config.yaml'
MODEL_PATH = ROOT_DIR / 'models' / 'saved_models' / 'best_model.pth'

sys.path.append(str(ROOT_DIR))
from src.model import PlantDiseaseModel

# Care tips and utilities
plant_care_tips = {
    "Corn_(maize)___healthy": {
        "short_term": [
            "Continue regular watering schedule",
            "Monitor for any changes in leaf color",
            "Maintain good air circulation"
        ],
        "long_term": [
            "Regular soil testing",
            "Crop rotation planning",
            "Preventive pest management"
        ]
    },
    "Tomato___Late_blight": {
        "short_term": [
            "Remove infected leaves immediately",
            "Apply appropriate fungicide",
            "Improve air circulation around plants"
        ],
        "long_term": [
            "Use resistant varieties next season",
            "Improve soil drainage",
            "Practice crop rotation"
        ]
    }
}

def load_model():
    """Load the trained model and class mappings"""
    try:
        # Load config
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        
        print(f"Loading model with {config['model']['num_classes']} classes")
        
        # Initialize model
        model = PlantDiseaseModel(num_classes=config['model']['num_classes'])
        
        # Load trained weights
        print(f"Loading checkpoint from: {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, checkpoint['class_to_idx']
        
    except Exception as e:
        print(f"Error in load_model: {str(e)}")
        raise e

def process_image(image):
    """Process and display image transformation steps"""
    st.write("🔍 Image Processing Steps:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        resized = transforms.Resize(256)(image)
        st.image(resized, caption="Resized Image")
        
    with col2:
        cropped = transforms.CenterCrop(224)(resized)
        st.image(cropped, caption="Cropped Image")
        
    with col3:
        st.write("Final Processing:")
        st.write("• Converted to tensor")
        st.write("• Normalized")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def main():
    st.set_page_config(page_title="Plant Disease Classifier", layout="wide")
    
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Create tabs
    tab1, tab2 = st.tabs(["Classifier", "Model Info"])
    
    with tab1:
        st.title("🌿 Plant Disease Classifier")
        
        try:
            model, class_to_idx = load_model()
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.write("Debugging info:")
            st.write(f"Config path exists: {CONFIG_PATH.exists()}")
            st.write(f"Model path exists: {MODEL_PATH.exists()}")
            return

        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption='Uploaded Image', use_column_width=True)
        
        with col2:
            if uploaded_file is not None:
                with st.spinner('Analyzing image...'):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)

                    try:
                        input_tensor = process_image(image)
                        
                        with torch.no_grad():
                            output = model(input_tensor)
                            probabilities = torch.nn.functional.softmax(output, dim=1)
                            predicted_idx = output.argmax(1).item()
                            confidence = probabilities[0][predicted_idx].item()
                        
                        predicted_class = idx_to_class[predicted_idx]
                        
                        # Store prediction in history
                        st.session_state.history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'prediction': predicted_class,
                            'confidence': confidence
                        })

                        # Show prediction and confidence
                        st.subheader("📊 Analysis Results")
                        if "healthy" in predicted_class.lower():
                            st.success(f"🌱 Prediction: {predicted_class.replace('_', ' ')}")
                        else:
                            st.warning(f"⚠️ Prediction: {predicted_class.replace('_', ' ')}")
                        
                        # Show class probabilities
                        for idx, prob in enumerate(probabilities[0]):
                            class_name = idx_to_class[idx].replace('_', ' ')
                            st.write(f"{class_name}: {prob*100:.2f}%")
                            st.progress(float(prob))

                        # Show care recommendations
                        if predicted_class in plant_care_tips:
                            st.subheader("🌱 Care Recommendations")
                            
                            col_short, col_long = st.columns(2)
                            with col_short:
                                st.write("Immediate Actions:")
                                for tip in plant_care_tips[predicted_class]["short_term"]:
                                    st.write(f"• {tip}")
                            with col_long:
                                st.write("Long-term Prevention:")
                                for tip in plant_care_tips[predicted_class]["long_term"]:
                                    st.write(f"• {tip}")
                    
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")

    with tab2:
        st.header("Model Architecture")
        st.write("""
        This classifier uses a ResNet50 architecture with transfer learning:
        - Pre-trained on ImageNet
        - Fine-tuned on plant disease dataset
        - 2 disease classes
        - Input size: 224x224 pixels
        """)
        
        st.subheader("Performance Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Training Accuracy', 'Validation Accuracy', 'Number of Parameters'],
            'Value': ['99.0%', '100%', '23.5M']
        })
        st.table(metrics_df)

    # Sidebar
    st.sidebar.title("Recent Predictions")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-5:]):
            st.sidebar.write(f"Time: {item['timestamp']}")
            st.sidebar.write(f"Prediction: {item['prediction'].replace('_', ' ')}")
            st.sidebar.write(f"Confidence: {item['confidence']*100:.2f}%")
            st.sidebar.divider()

if __name__ == "__main__":
    main()