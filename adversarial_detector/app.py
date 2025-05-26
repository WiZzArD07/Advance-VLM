import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set page config
st.set_page_config(
    page_title="Adversarial Image Generator",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .image-container {
        display: flex;
        justify-content: space-between;
        margin: 20px 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def load_image(image_file):
    """Load and preprocess an image."""
    try:
        image = Image.open(image_file).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0), image
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None, None

def fgsm_attack(image, epsilon, data_grad):
    """Create FGSM adversarial example."""
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def pgd_attack(model, image, label, epsilon=0.03, alpha=0.01, num_iter=40):
    """Create PGD adversarial example."""
    perturbed_image = image.clone().detach()
    
    for _ in range(num_iter):
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        loss = nn.CrossEntropyLoss()(output, label)
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            perturbed_image = perturbed_image + alpha * perturbed_image.grad.sign()
            eta = torch.clamp(perturbed_image - image, -epsilon, epsilon)
            perturbed_image = torch.clamp(image + eta, 0, 1).detach()
    
    return perturbed_image

def tensor_to_image(tensor):
    """Convert tensor to PIL Image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = tensor * std + mean
    image = transforms.ToPILImage()(tensor.squeeze(0))
    return image

def get_class_name(class_idx):
    """Get ImageNet class name from index."""
    with open('imagenet_classes.txt') as f:
        categories = [s.strip() for s in f.readlines()]
    return categories[class_idx]

def main():
    st.title("🎨 Adversarial Image Generator")
    st.markdown("""
    This application generates adversarial examples from your input images using FGSM and PGD attacks.
    Upload an image and adjust the parameters to see how different attacks affect the image.
    """)

    # Sidebar for parameters
    st.sidebar.header("Attack Parameters")
    
    # Attack Type Selection
    attack_type = st.sidebar.radio(
        "Select Attack Type",
        ["FGSM", "PGD", "Both"],
        help="Choose which attack(s) to generate"
    )
    
    # FGSM Parameters
    st.sidebar.subheader("FGSM Attack")
    fgsm_epsilon = st.sidebar.slider(
        "FGSM Epsilon",
        0.0, 0.1, 0.03, 0.001,
        help="Step size for FGSM attack. Higher values create more visible perturbations."
    )
    
    # PGD Parameters
    st.sidebar.subheader("PGD Attack")
    pgd_epsilon = st.sidebar.slider(
        "PGD Epsilon",
        0.0, 0.1, 0.03, 0.001,
        help="Maximum perturbation size. Higher values allow for larger changes."
    )
    pgd_alpha = st.sidebar.slider(
        "PGD Alpha",
        0.001, 0.05, 0.01, 0.001,
        help="Step size for each iteration. Smaller values create more subtle changes."
    )
    pgd_iterations = st.sidebar.slider(
        "PGD Iterations",
        10, 100, 40, 10,
        help="Number of iterations. More iterations = stronger attack."
    )

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load model
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.eval()
        
        # Load and preprocess image
        image_tensor, original_image = load_image(uploaded_file)
        
        if image_tensor is not None:
            # Get original prediction
            with torch.no_grad():
                output = model(image_tensor)
                _, predicted = torch.max(output, 1)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                top_prob, top_class = torch.max(probabilities, 1)
            
            # Display original image and prediction
            st.subheader("Original Image")
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_image, use_column_width=True)
            with col2:
                st.write("### Prediction Details")
                st.write(f"**Class:** {get_class_name(predicted.item())}")
                st.write(f"**Confidence:** {top_prob.item():.2%}")
                
                # Show top 3 predictions
                top3_prob, top3_idx = torch.topk(probabilities, 3)
                st.write("### Top 3 Predictions:")
                for prob, idx in zip(top3_prob[0], top3_idx[0]):
                    st.write(f"- {get_class_name(idx.item())}: {prob.item():.2%}")
            
            # Generate and display adversarial examples
            if attack_type in ["FGSM", "Both"]:
                st.subheader("FGSM Adversarial Example")
                # Generate FGSM adversarial example
                image_tensor.requires_grad = True
                output = model(image_tensor)
                loss = nn.CrossEntropyLoss()(output, predicted)
                model.zero_grad()
                loss.backward()
                fgsm_image = fgsm_attack(image_tensor, fgsm_epsilon, image_tensor.grad.data)
                
                # Get FGSM predictions
                with torch.no_grad():
                    fgsm_output = model(fgsm_image)
                    fgsm_prob = torch.nn.functional.softmax(fgsm_output, dim=1)
                    fgsm_top_prob, fgsm_top_class = torch.max(fgsm_prob, 1)
                
                col3, col4 = st.columns(2)
                with col3:
                    st.image(tensor_to_image(fgsm_image), use_column_width=True)
                with col4:
                    st.write("### FGSM Attack Results")
                    st.write(f"**New Class:** {get_class_name(fgsm_top_class.item())}")
                    st.write(f"**Confidence:** {fgsm_top_prob.item():.2%}")
                    st.write(f"**Attack Strength (Epsilon):** {fgsm_epsilon}")
                    
                    # Show top 3 predictions
                    top3_prob, top3_idx = torch.topk(fgsm_prob, 3)
                    st.write("### Top 3 Predictions:")
                    for prob, idx in zip(top3_prob[0], top3_idx[0]):
                        st.write(f"- {get_class_name(idx.item())}: {prob.item():.2%}")
            
            if attack_type in ["PGD", "Both"]:
                st.subheader("PGD Adversarial Example")
                # Generate PGD adversarial example
                pgd_image = pgd_attack(model, image_tensor, predicted, 
                                     epsilon=pgd_epsilon, 
                                     alpha=pgd_alpha, 
                                     num_iter=pgd_iterations)
                
                # Get PGD predictions
                with torch.no_grad():
                    pgd_output = model(pgd_image)
                    pgd_prob = torch.nn.functional.softmax(pgd_output, dim=1)
                    pgd_top_prob, pgd_top_class = torch.max(pgd_prob, 1)
                
                col5, col6 = st.columns(2)
                with col5:
                    st.image(tensor_to_image(pgd_image), use_column_width=True)
                with col6:
                    st.write("### PGD Attack Results")
                    st.write(f"**New Class:** {get_class_name(pgd_top_class.item())}")
                    st.write(f"**Confidence:** {pgd_top_prob.item():.2%}")
                    st.write(f"**Attack Parameters:**")
                    st.write(f"- Epsilon: {pgd_epsilon}")
                    st.write(f"- Alpha: {pgd_alpha}")
                    st.write(f"- Iterations: {pgd_iterations}")
                    
                    # Show top 3 predictions
                    top3_prob, top3_idx = torch.topk(pgd_prob, 3)
                    st.write("### Top 3 Predictions:")
                    for prob, idx in zip(top3_prob[0], top3_idx[0]):
                        st.write(f"- {get_class_name(idx.item())}: {prob.item():.2%}")
            
            # Display attack statistics
            st.subheader("Attack Statistics")
            col7, col8, col9 = st.columns(3)
            
            with col7:
                st.metric("Original Confidence", f"{top_prob.item():.2%}")
            if attack_type in ["FGSM", "Both"]:
                with col8:
                    st.metric("FGSM Confidence", f"{fgsm_top_prob.item():.2%}")
            if attack_type in ["PGD", "Both"]:
                with col9:
                    st.metric("PGD Confidence", f"{pgd_top_prob.item():.2%}")
            
            # Add explanation box
            st.markdown("""
            ### Understanding the Results
            
            **Original Image:**
            - Shows the model's prediction on the clean image
            - Higher confidence indicates the model is more certain
            
            **FGSM Attack:**
            - Quick, single-step attack
            - Creates visible noise patterns
            - Good for testing basic model robustness
            
            **PGD Attack:**
            - More sophisticated, iterative attack
            - Creates subtler perturbations
            - Better at fooling the model
            - Takes longer to compute
            
            **Interpreting Results:**
            - Successful attack: Different prediction with lower confidence
            - Failed attack: Same prediction or high confidence
            - Adjust parameters to find the right balance between attack strength and subtlety
            """)

if __name__ == "__main__":
    main() 