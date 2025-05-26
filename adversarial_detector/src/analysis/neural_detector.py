import numpy as np
import cv2
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class NeuralDetector:
    def __init__(self, config):
        self.config = config
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the detector with baseline features."""
        # Create synthetic normal image features for training
        normal_features = self._generate_normal_features(100)
        self.scaler.fit(normal_features)
        self.isolation_forest.fit(self.scaler.transform(normal_features))
    
    def analyze(self, image):
        """Neural network-based adversarial detection."""
        result = {
            'score': 0.0,
            'detected': False,
            'anomaly_score': 0.0,
            'feature_analysis': {}
        }
        
        # Extract features
        features = self._extract_features(image)
        
        # Normalize features
        features_scaled = self.scaler.transform([features])
        
        # Get anomaly score
        anomaly_score = self.isolation_forest.decision_function(features_scaled)[0]
        is_outlier = self.isolation_forest.predict(features_scaled)[0] == -1
        
        # Convert anomaly score to [0, 1] range
        normalized_score = max(0, min(1, (0.5 - anomaly_score) * 2))
        
        result['score'] = normalized_score
        result['detected'] = is_outlier
        result['anomaly_score'] = anomaly_score
        result['feature_analysis'] = self._analyze_features(features)
        
        return result
    
    def _extract_features(self, image):
        """Extract comprehensive features from image."""
        features = []
        
        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 1. Statistical features
        for channel in [image[:,:,0], image[:,:,1], image[:,:,2], gray]:
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.min(channel),
                np.max(channel),
                np.median(channel)
            ])
        
        # 2. Texture features (Local Binary Pattern inspired)
        texture_features = self._extract_texture_features(gray)
        features.extend(texture_features)
        
        # 3. Edge features
        edges = cv2.Canny(gray, 50, 150)
        features.extend([
            np.sum(edges > 0) / edges.size,  # Edge density
            np.mean(edges),
            np.std(edges)
        ])
        
        # 4. Frequency domain features
        fft_features = self._extract_fft_features(gray)
        features.extend(fft_features)
        
        # 5. Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([
            np.mean(grad_mag),
            np.std(grad_mag),
            np.max(grad_mag)
        ])
        
        return np.array(features)
    
    def _extract_texture_features(self, gray_image):
        """Extract texture-based features."""
        features = []
        
        # Local standard deviation
        kernel = np.ones((3, 3)) / 9
        local_mean = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
        local_std = np.sqrt(cv2.filter2D((gray_image.astype(np.float32) - local_mean)**2, -1, kernel))
        
        features.extend([
            np.mean(local_std),
            np.std(local_std),
            np.max(local_std)
        ])
        
        # Contrast features
        contrast = cv2.Laplacian(gray_image, cv2.CV_64F)
        features.extend([
            np.mean(np.abs(contrast)),
            np.std(contrast)
        ])
        
        return features
    
    def _extract_fft_features(self, gray_image):
        """Extract frequency domain features."""
        # Compute FFT
        fft = np.fft.fft2(gray_image)
        magnitude = np.abs(fft)
        
        # Features from magnitude spectrum
        features = [
            np.mean(magnitude),
            np.std(magnitude),
            np.max(magnitude)
        ]
        
        # High frequency energy
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        high_freq_mask = np.zeros((h, w))
        high_freq_mask[center_h//2:center_h+center_h//2, center_w//2:center_w+center_w//2] = 1
        high_freq_mask = 1 - high_freq_mask  # Invert to get high frequencies
        
        high_freq_energy = np.sum(magnitude * high_freq_mask)
        total_energy = np.sum(magnitude)
        
        if total_energy > 0:
            features.append(high_freq_energy / total_energy)
        else:
            features.append(0.0)
        
        return features
    
    def _generate_normal_features(self, num_samples):
        """Generate synthetic normal image features for training."""
        normal_features = []
        
        for _ in range(num_samples):
            # Create synthetic normal image
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            
            # Add some natural structure
            img = cv2.GaussianBlur(img, (5, 5), 1.0)
            
            features = self._extract_features(img)
            normal_features.append(features)
        
        return np.array(normal_features)
    
    def _analyze_features(self, features):
        """Analyze extracted features for interpretation."""
        analysis = {
            'statistical_anomalies': [],
            'texture_anomalies': [],
            'frequency_anomalies': []
        }
        
        # Simple thresholding for interpretation
        if len(features) > 20:
            # Check statistical features (first 20)
            stat_features = features[:20]
            if np.any(stat_features > 200) or np.any(stat_features < 10):
                analysis['statistical_anomalies'].append("Unusual pixel value distribution")
            
            # Check texture features
            if len(features) > 25:
                texture_features = features[20:25]
                if np.any(texture_features > 50):
                    analysis['texture_anomalies'].append("High texture variation detected")
            
            # Check frequency features
            if len(features) > 30:
                freq_features = features[25:30]
                if np.any(freq_features > 1000):
                    analysis['frequency_anomalies'].append("Unusual frequency characteristics")
        
        return analysis