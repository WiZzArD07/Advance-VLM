import cv2
import numpy as np
from PIL import Image
import os

class ImageProcessor:
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    def load_image(self, image_path):
        """Load image from file path."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Check file extension
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in self.supported_formats:
                raise ValueError(f"Unsupported image format: {ext}")
            
            # Load image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                # Try with PIL as fallback
                pil_image = Image.open(image_path)
                image = np.array(pil_image)
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Convert BGR to RGB for consistency
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
            
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            return None
    
    def preprocess_image(self, image, target_size=None):
        """Preprocess image for analysis."""
        if image is None:
            return None
        
        processed = image.copy()
        
        # Resize if target size specified
        if target_size:
            processed = cv2.resize(processed, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Ensure uint8 format
        if processed.dtype != np.uint8:
            processed = np.clip(processed, 0, 255).astype(np.uint8)
        
        return processed
    
    def save_image(self, image, output_path):
        """Save image to file."""
        try:
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert RGB to BGR for OpenCV saving
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, image_bgr)
            else:
                cv2.imwrite(output_path, image)
            return True
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return False