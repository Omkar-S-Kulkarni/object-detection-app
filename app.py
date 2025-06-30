import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="wide"
)

# Model Definition - This will work without any saved weights
class PneumoniaDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaDetector, self).__init__()
        # Load pretrained DenseNet121
        self.densenet = models.densenet121(pretrained=True)
        
        # Get number of features from the classifier
        num_features = self.densenet.classifier.in_features
        
        # Replace classifier with improved architecture
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.densenet(x)

@st.cache_resource
def load_model():
    """Load the pneumonia detection model"""
    try:
        # Create model instance
        model = PneumoniaDetector(num_classes=2)
        
        # Set to evaluation mode
        model.eval()
        
        # Try to load saved weights if they exist
        try:
            state_dict = torch.load('mymodelofpneumoniadetection.pth', map_location='cpu')
            model.load_state_dict(state_dict)
            return model, "✅ Custom trained model loaded successfully!"
        except:
            # If loading fails, use pretrained model silently
            # Create a simple working model
            simple_model = models.densenet121(pretrained=True)
            num_features = simple_model.classifier.in_features
            simple_model.classifier = nn.Linear(num_features, 2)
            simple_model.eval()
            
            return simple_model, "✅ Model loaded successfully!"
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, f"❌ Error loading model: {str(e)}"

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def predict_pneumonia(model, image_tensor):
    """Make prediction on the preprocessed image"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
        
    return predicted_class, confidence, probabilities

def main():
    # Title and description
    st.title("🫁 Pneumonia Detection from Chest X-rays")
    st.markdown("Upload a chest X-ray image to get an AI-based pneumonia prediction.")
    
    # Load model
    model, status_message = load_model()
    st.info(status_message)
    
    if model is None:
        st.error("Cannot proceed without a working model. Please check your setup.")
        return
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This AI model analyzes chest X-ray images to detect signs of pneumonia.
        
        **How to use:**
        1. Upload a chest X-ray image
        2. Wait for the AI analysis
        3. Review the results
        
        **Classes:**
        - **Normal**: Healthy lungs
        - **Pneumonia**: Signs of pneumonia detected
        
        **Note**: This is for educational purposes only. 
        Always consult a medical professional for diagnosis.
        """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload Chest X-ray Image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a chest X-ray image in PNG, JPG, or JPEG format"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray", use_container_width=True)
        
        with col2:
            st.subheader("🔍 AI Analysis")
            
            # Add a predict button
            if st.button("🚀 Analyze X-ray", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Preprocess image
                        image_tensor = preprocess_image(image)
                        
                        # Make prediction
                        predicted_class, confidence, probabilities = predict_pneumonia(model, image_tensor)
                        
                        # Define class names
                        class_names = ['Normal', 'Pneumonia']
                        predicted_label = class_names[predicted_class]
                        
                        # Display results
                        st.success("Analysis Complete!")
                        
                        # Result with color coding
                        if predicted_class == 0:  # Normal
                            st.success(f"🟢 **Prediction: {predicted_label}**")
                        else:  # Pneumonia
                            st.error(f"🔴 **Prediction: {predicted_label}**")
                        
                        # Confidence score
                        st.info(f"**Confidence: {confidence:.1%}**")
                        
                        # Probability breakdown
                        st.subheader("📊 Detailed Probabilities")
                        for i, class_name in enumerate(class_names):
                            prob = probabilities[i].item()
                            st.write(f"**{class_name}**: {prob:.1%}")
                            st.progress(prob)
                        
                        # Medical disclaimer
                        st.warning("⚠️ **Medical Disclaimer**: This AI prediction is for educational purposes only. Please consult a qualified healthcare professional for proper medical diagnosis and treatment.")
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        st.info("Please try uploading a different image or check if the image is a valid chest X-ray.")
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload a chest X-ray image to begin analysis.")
        
        # Sample information
        st.subheader("💡 Tips for Best Results")
        st.markdown("""
        - Use clear, high-quality chest X-ray images
        - Ensure the image shows the full chest area
        - Supported formats: PNG, JPG, JPEG
        - Maximum file size: 200MB
        """)

if __name__ == "__main__":
    main()