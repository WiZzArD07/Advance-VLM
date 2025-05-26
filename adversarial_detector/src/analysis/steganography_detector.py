import numpy as np
import cv2
from collections import Counter
import re

class SteganographyDetector:
    def __init__(self, config):
        self.config = config
        self.threshold = config['detection']['steganography_threshold']
        self.channels = config['analysis']['lsb_channels']
    
    def analyze(self, image):
        """Detect hidden content using steganography techniques."""
        result = {
            'score': 0.0,
            'detected': False,
            'hidden_text': [],
            'stego_analysis': {}
        }
        
        # 1. LSB Analysis
        lsb_score, hidden_text = self._analyze_lsb(image)
        
        # 2. Statistical Analysis
        statistical_score = self._statistical_analysis(image)
        
        # 3. Visual Attack Detection
        visual_score = self._detect_visual_attacks(image)
        
        # Combine scores
        result['score'] = np.mean([lsb_score, statistical_score, visual_score])
        result['detected'] = result['score'] > self.threshold
        result['hidden_text'] = hidden_text
        
        result['stego_analysis'] = {
            'lsb_score': lsb_score,
            'statistical_score': statistical_score,
            'visual_attack_score': visual_score
        }
        
        return result
    
    def _analyze_lsb(self, image):
        """Analyze Least Significant Bit patterns."""
        hidden_texts = []
        total_score = 0.0
        
        for channel_idx, channel_name in enumerate(['R', 'G', 'B']):
            if channel_name in self.channels:
                channel_data = image[:, :, channel_idx]
                
                # Extract LSBs
                lsb_data = channel_data & 1
                lsb_flat = lsb_data.flatten()
                
                # Convert LSB sequence to potential text
                text = self._lsb_to_text(lsb_flat)
                if text:
                    hidden_texts.extend(text)
                
                # Statistical analysis of LSB distribution
                lsb_entropy = self._calculate_entropy(lsb_flat)
                total_score += lsb_entropy
        
        avg_score = total_score / len(self.channels) if self.channels else 0.0
        return min(avg_score, 1.0), hidden_texts
    
    def _lsb_to_text(self, lsb_sequence):
        """Convert LSB sequence to potential hidden text."""
        texts = []
        
        # Try to extract ASCII text
        for start_idx in range(0, min(len(lsb_sequence), 1000), 8):
            if start_idx + 8 <= len(lsb_sequence):
                # Convert 8 LSBs to byte
                byte_value = 0
                for i in range(8):
                    byte_value |= (lsb_sequence[start_idx + i] << i)
                
                # Check if it's printable ASCII
                if 32 <= byte_value <= 126:
                    char = chr(byte_value)
                    if len(texts) == 0 or texts[-1][-1] != char:
                        if len(texts) == 0:
                            texts.append(char)
                        else:
                            texts[-1] += char
                else:
                    if texts and len(texts[-1]) > 0:
                        texts.append("")
        
        # Filter out short or non-meaningful sequences
        meaningful_texts = []
        for text in texts:
            if len(text) >= 4 and self._is_meaningful_text(text):
                meaningful_texts.append(f"LSB Hidden Text: '{text}'")
        
        return meaningful_texts[:5]  # Limit to 5 findings
    
    def _is_meaningful_text(self, text):
        """Check if extracted text appears meaningful."""
        # Check for common patterns that might indicate prompts or instructions
        jailbreak_patterns = [
            r'ignore.*instruction', r'forget.*rule', r'act.*as', 
            r'pretend.*to.*be', r'role.*play', r'system.*prompt',
            r'override.*safety', r'bypass.*filter'
        ]
        
        text_lower = text.lower()
        for pattern in jailbreak_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Check for reasonable character distribution
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        return alpha_ratio > 0.5 and len(text.strip()) > 3
    
    def _statistical_analysis(self, image):
        """Statistical analysis for steganography detection."""
        scores = []
        
        for channel_idx in range(3):
            channel = image[:, :, channel_idx]
            
            # Chi-square test for LSB randomness
            lsb = channel & 1
            lsb_hist = Counter(lsb.flatten())
            
            # Expected uniform distribution
            total_pixels = lsb.size
            expected_freq = total_pixels / 2
            
            # Chi-square statistic
            chi_square = 0
            for value in [0, 1]:
                observed = lsb_hist.get(value, 0)
                chi_square += ((observed - expected_freq) ** 2) / expected_freq
            
            # Normalize chi-square to [0, 1]
            normalized_chi_square = min(chi_square / 10.0, 1.0)
            scores.append(normalized_chi_square)
        
        return np.mean(scores)
    
    def _detect_visual_attacks(self, image):
        """Detect visual prompt injection attacks."""
        # Convert to grayscale for text detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge detection to find potential text regions
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours that might represent text
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contour characteristics
        text_like_regions = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 10 < area < 1000:  # Reasonable size for text
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Text-like aspect ratio
                if 0.1 < aspect_ratio < 10:
                    text_like_regions += 1
        
        # Score based on density of text-like regions
        image_area = image.shape[0] * image.shape[1]
        text_density = text_like_regions / (image_area / 10000)  # Normalize by image size
        
        return min(text_density, 1.0)
    
    def _calculate_entropy(self, data):
        """Calculate Shannon entropy."""
        unique_values, counts = np.unique(data, return_counts=True)
        probabilities = counts / len(data)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-8))
        return entropy
