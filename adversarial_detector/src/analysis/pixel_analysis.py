import numpy as np
import cv2
from scipy import stats

class PixelAnalyzer:
    def __init__(self, config):
        self.config = config
        self.threshold = config['detection']['pixel_threshold']
    
    def analyze(self, image):
        """Analyze pixel-level anomalies."""
        result = {
            'score': 0.0,
            'detected': False,
            'anomalies': [],
            'statistics': {}
        }
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # 1. Statistical anomaly detection
        rgb_stats = self._compute_channel_statistics(image)
        hsv_stats = self._compute_channel_statistics(hsv)
        
        # 2. Noise pattern analysis
        noise_score = self._detect_adversarial_noise(image)
        
        # 3. Gradient analysis
        gradient_score = self._analyze_gradients(image)
        
        # 4. Local texture analysis
        texture_score = self._analyze_texture_anomalies(image)
        
        # Combine scores
        scores = [noise_score, gradient_score, texture_score]
        result['score'] = np.mean(scores)
        result['detected'] = result['score'] > self.threshold
        
        result['statistics'] = {
            'rgb_stats': rgb_stats,
            'hsv_stats': hsv_stats,
            'noise_score': noise_score,
            'gradient_score': gradient_score,
            'texture_score': texture_score
        }
        
        return result
    
    def _compute_channel_statistics(self, image):
        """Compute statistical properties of image channels."""
        stats_dict = {}
        for i, channel in enumerate(['ch1', 'ch2', 'ch3']):
            channel_data = image[:, :, i].flatten()
            stats_dict[channel] = {
                'mean': np.mean(channel_data),
                'std': np.std(channel_data),
                'skewness': stats.skew(channel_data),
                'kurtosis': stats.kurtosis(channel_data),
                'entropy': self._calculate_entropy(channel_data)
            }
        return stats_dict
    
    def _calculate_entropy(self, data):
        """Calculate Shannon entropy of data."""
        hist, _ = np.histogram(data, bins=256, range=(0, 255))
        hist = hist[hist > 0]  # Remove zero entries
        hist = hist / np.sum(hist)  # Normalize
        return -np.sum(hist * np.log2(hist))
    
    def _detect_adversarial_noise(self, image):
        """Detect adversarial noise patterns."""
        # High-frequency noise detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply high-pass filter
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filtered = cv2.filter2D(gray, -1, kernel)
        
        # Calculate noise energy
        noise_energy = np.mean(np.abs(filtered))
        
        # Normalize to [0, 1]
        return min(noise_energy / 50.0, 1.0)
    
    def _analyze_gradients(self, image):
        """Analyze gradient patterns for adversarial perturbations."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Analyze gradient distribution
        grad_std = np.std(grad_mag)
        grad_mean = np.mean(grad_mag)
        
        # High gradient variance might indicate adversarial perturbations
        gradient_score = min(grad_std / (grad_mean + 1e-8) / 2.0, 1.0)
        
        return gradient_score
    
    def _analyze_texture_anomalies(self, image):
        """Analyze local texture patterns."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Local Binary Pattern-like analysis
        h, w = gray.shape
        texture_score = 0.0
        
        # Sample patches
        patch_size = 8
        num_patches = min(100, (h // patch_size) * (w // patch_size))
        
        for _ in range(num_patches):
            y = np.random.randint(0, h - patch_size)
            x = np.random.randint(0, w - patch_size)
            
            patch = gray[y:y+patch_size, x:x+patch_size]
            
            # Calculate local variance
            local_var = np.var(patch)
            texture_score += local_var
        
        # Normalize
        texture_score = min(texture_score / (num_patches * 1000), 1.0)
        
        return texture_score
