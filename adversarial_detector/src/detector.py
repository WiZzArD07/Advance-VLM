import numpy as np
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from analysis.pixel_analysis import PixelAnalyzer
from analysis.frequency_analysis import FrequencyAnalyzer
from analysis.steganography_detector import SteganographyDetector
from analysis.neural_detector import NeuralDetector
from utils.image_processor import ImageProcessor

class AdversarialDetector:
    def __init__(self, config):
        self.config = config
        self.pixel_analyzer = PixelAnalyzer(config)
        self.frequency_analyzer = FrequencyAnalyzer(config)
        self.stego_detector = SteganographyDetector(config)
        self.neural_detector = NeuralDetector(config)
        self.image_processor = ImageProcessor()
        
    def detect_adversarial(self, image_path, verbose=False):
        """Main detection pipeline."""
        try:
            # Load and preprocess image
            image = self.image_processor.load_image(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            results = {
                'image_path': image_path,
                'image_shape': image.shape,
                'detailed_results': {},
                'hidden_content': [],
                'overall_score': 0.0,
                'is_adversarial': False
            }
            
            if verbose:
                print("Running pixel-level analysis...")
            pixel_result = self.pixel_analyzer.analyze(image)
            results['detailed_results']['pixel_analysis'] = pixel_result
            
            if verbose:
                print("Running frequency domain analysis...")
            freq_result = self.frequency_analyzer.analyze(image)
            results['detailed_results']['frequency_analysis'] = freq_result
            
            if verbose:
                print("Running steganography detection...")
            stego_result = self.stego_detector.analyze(image)
            results['detailed_results']['steganography'] = stego_result
            results['hidden_content'].extend(stego_result.get('hidden_text', []))
            
            if verbose:
                print("Running neural network detection...")
            neural_result = self.neural_detector.analyze(image)
            results['detailed_results']['neural_detection'] = neural_result
            
            # Ensemble scoring
            scores = [
                pixel_result['score'],
                freq_result['score'],
                stego_result['score'],
                neural_result['score']
            ]
            
            # Weighted ensemble
            weights = [0.25, 0.30, 0.25, 0.20]
            results['overall_score'] = sum(w * s for w, s in zip(weights, scores))
            results['is_adversarial'] = results['overall_score'] > self.config['detection']['ensemble_threshold']
            
            return results
            
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            return None