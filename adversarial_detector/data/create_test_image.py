from PIL import Image, ImageDraw
import os

def create_test_image(output_path):
    # Create a 500x500 white image
    image = Image.new('RGB', (500, 500), 'white')
    draw = ImageDraw.Draw(image)
    
    # Draw some shapes to make it interesting
    # Draw a red circle
    draw.ellipse([100, 100, 400, 400], fill='red')
    
    # Draw a blue rectangle
    draw.rectangle([150, 150, 350, 350], fill='blue')
    
    # Draw a green triangle
    draw.polygon([(250, 100), (400, 300), (100, 300)], fill='green')
    
    # Save the image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    print(f"Test image created at: {output_path}")

if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "samples", "sample_image.jpg")
    create_test_image(output_path) 