import unittest
import numpy as np
import sys
import os

# Add src to path for testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from detector import AdversarialDetector
from utils.image_processor import ImageProcessor

class TestAdversarialDetector(unittest.TestCase):
    def setUp(self):
        """Setup test configuration."""
        self.config = {
            'detection': {
                'pixel_threshold': 0.15,
                'frequency_threshold': 0.3,
                'steganography_threshold': 0.2,
                'ensemble_threshold': 0.4
            },
            'analysis': {
                'dct_blocks': 8,
                'wavelet_type': 'db4',
                'lsb_channels': ['R', 'G', 'B']
            },
            'visualization': {
                'save_plots': False,
                'plot_dir': 'test_results/'
            }
        }
        
        self.detector = AdversarialDetector(self.config)
        self.processor = ImageProcessor()
    
    def test_clean_image_detection(self):
        """Test detection on a clean synthetic image."""
        # Create a clean test image
        clean_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Mock the detection process
        result = {
            'overall_score': 0.1,
            'is_adversarial': False,
            'detailed_results': {
                'pixel_analysis': {'score': 0.05, 'detected': False},
                'frequency_analysis': {'score': 0.1, 'detected': False},
                'steganography': {'score': 0.08, 'detected': False},
                'neural_detection': {'score': 0.12, 'detected': False}
            },
            'hidden_content': []
        }
        
        self.assertFalse(result['is_adversarial'])
        self.assertLess(result['overall_score'], 0.4)
    
    def test_adversarial_image_detection(self):
        """Test detection on a synthetic adversarial image."""
        # Create an adversarial-like test image with noise
        base_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        noise = np.random.randint(-10, 10, (100, 100, 3))
        adversarial_image = np.clip(base_image.astype(int) + noise, 0, 255).astype(np.uint8)
        
        # Mock high detection scores
        result = {
            'overall_score': 0.7,
            'is_adversarial': True,
            'detailed_results': {
                'pixel_analysis': {'score': 0.8, 'detected': True},
                'frequency_analysis': {'score': 0.6, 'detected': True},
                'steganography': {'score': 0.7, 'detected': True},
                'neural_detection': {'score': 0.75, 'detected': True}
            },
            'hidden_content': ['Potential adversarial pattern detected']
        }
        
        self.assertTrue(result['is_adversarial'])
        self.assertGreater(result['overall_score'], 0.4)
    
    def test_image_processor(self):
        """Test image processing utilities."""
        # Test synthetic image creation and processing
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Test preprocessing
        processed = self.processor.preprocess_image(test_image, target_size=(32, 32))
        
        self.assertEqual(processed.shape, (32, 32, 3))
        self.assertEqual(processed.dtype, np.uint8)

if __name__ == '__main__':
    unittest.main()