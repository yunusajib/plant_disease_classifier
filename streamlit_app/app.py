# streamlit_app/app.py
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import yaml
from pathlib import Path
import sys
import os

# Add project root to path and get absolute paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = ROOT_DIR / 'configs' / 'config.yaml'
MODEL_PATH = ROOT_DIR / 'models' / 'saved_models' / 'best_model.pth'

sys.path.append(str(ROOT_DIR))
from src.model import PlantDiseaseModel

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
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def main():
    st.set_page_config(page_title="Plant Disease Classifier", layout="wide")
    
    st.title("🌿 Plant Disease Classifier")
    st.write("Upload an image of a corn or tomato leaf to check its health status")
    
    # Load model with error handling
    try:
        model, class_to_idx = load_model()
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        print(f"Available classes: {list(class_to_idx.keys())}")
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.write("Debugging info:")
        st.write(f"Config path exists: {CONFIG_PATH.exists()}")
        st.write(f"Model path exists: {MODEL_PATH.exists()}")
        return
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        if uploaded_file is not None:
            try:
                # Process image and get prediction
                input_tensor = process_image(image)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    predicted_idx = output.argmax(1).item()
                    confidence = probabilities[0][predicted_idx].item()
                
                predicted_class = idx_to_class[predicted_idx]
                
                # Show prediction with appropriate styling
                if "healthy" in predicted_class.lower():
                    st.success(f"🌱 Prediction: {predicted_class.replace('_', ' ')}")
                else:
                    st.warning(f"⚠️ Prediction: {predicted_class.replace('_', ' ')}")
                
                st.info(f"Confidence: {confidence*100:.1f}%")
                
                # Add care recommendations
                st.subheader("Recommendations:")
                if "healthy" in predicted_class.lower():
                    st.write("""
                    ✅ Plant appears healthy! Continue with:
                    - Regular watering
                    - Proper fertilization
                    - Regular monitoring
                    """)
                else:
                    st.write("""
                    🚨 Possible disease detected! Recommended actions:
                    - Isolate affected plants
                    - Remove infected leaves
                    - Improve air circulation
                    - Consider appropriate fungicide
                    - Monitor surrounding plants
                    """)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.write("Please try another image")

    # Add sidebar information
    st.sidebar.title("About")
    st.sidebar.info("""
    This classifier can detect:
    - Healthy Corn Leaves
    - Tomato Late Blight
    
    Model Performance:
    - Training Accuracy: 99.0%
    - Validation Accuracy: 100%
    """)

if __name__ == "__main__":
    main()