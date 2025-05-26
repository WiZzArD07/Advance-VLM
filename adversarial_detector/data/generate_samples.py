import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import sys

def load_image(image_path):
    """Load and preprocess an image."""
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        print("Please provide a valid image path.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        sys.exit(1)

def save_image(tensor, filename):
    """Save a tensor as an image."""
    try:
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = tensor * std + mean
        
        # Convert to PIL Image and save
        image = transforms.ToPILImage()(tensor.squeeze(0))
        image.save(filename)
        print(f"Saved image to {filename}")
    except Exception as e:
        print(f"Error saving image: {str(e)}")

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

def generate_adversarial_samples(input_image_path, output_dir):
    """Generate various types of adversarial samples."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.eval()
        
        # Load and preprocess image
        image = load_image(input_image_path)
        
        # Get original prediction
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
        
        # Generate FGSM adversarial example
        image.requires_grad = True
        output = model(image)
        loss = nn.CrossEntropyLoss()(output, predicted)
        model.zero_grad()
        loss.backward()
        fgsm_image = fgsm_attack(image, epsilon=0.03, data_grad=image.grad.data)
        save_image(fgsm_image, os.path.join(output_dir, 'fgsm_adversarial.png'))
        
        # Generate PGD adversarial example
        pgd_image = pgd_attack(model, image, predicted)
        save_image(pgd_image, os.path.join(output_dir, 'pgd_adversarial.png'))
        
        print(f"Adversarial samples generated in {output_dir}")
        
    except Exception as e:
        print(f"Error generating adversarial samples: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths relative to the script location
    input_image = os.path.join(current_dir, "samples", "sample_image.jpg")
    output_directory = os.path.join(current_dir, "adversarial_samples")
    
    print(f"Looking for input image at: {input_image}")
    print(f"Output will be saved to: {output_directory}")
    
    if not os.path.exists(input_image):
        print("\nError: Input image not found!")
        print("Please place a sample image named 'sample_image.jpg' in the following directory:")
        print(f"{os.path.join(current_dir, 'samples')}")
        print("\nYou can use any JPG image for testing.")
        sys.exit(1)
    
    generate_adversarial_samples(input_image, output_directory) 