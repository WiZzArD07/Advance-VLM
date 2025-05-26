import numpy as np
import cv2
import pywt
from scipy.fft import fft2, fftshift

class FrequencyAnalyzer:
    def __init__(self, config):
        self.config = config
        self.threshold = config['detection']['frequency_threshold']
        self.dct_block_size = config['analysis']['dct_blocks']
        self.wavelet_type = config['analysis']['wavelet_type']
    
    def analyze(self, image):
        """Analyze frequency domain characteristics."""
        result = {
            'score': 0.0,
            'detected': False,
            'frequency_anomalies': [],
            'analysis': {}
        }
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 1. DCT Analysis
        dct_score = self._analyze_dct(gray)
        
        # 2. Wavelet Analysis
        wavelet_score = self._analyze_wavelets(gray)
        
        # 3. FFT Analysis
        fft_score = self._analyze_fft(gray)
        
        # Combine scores
        result['score'] = np.mean([dct_score, wavelet_score, fft_score])
        result['detected'] = result['score'] > self.threshold
        
        result['analysis'] = {
            'dct_score': dct_score,
            'wavelet_score': wavelet_score,
            'fft_score': fft_score
        }
        
        return result
    
    def _analyze_dct(self, image):
        """Analyze DCT coefficients for anomalies."""
        h, w = image.shape
        block_size = self.dct_block_size
        
        dct_scores = []
        
        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block = image[y:y+block_size, x:x+block_size].astype(np.float32)
                
                # Apply DCT
                dct_block = cv2.dct(block)
                
                # Analyze high-frequency coefficients
                high_freq = dct_block[block_size//2:, block_size//2:]
                high_freq_energy = np.sum(np.abs(high_freq))
                
                # Normalize by total energy
                total_energy = np.sum(np.abs(dct_block))
                if total_energy > 0:
                    dct_scores.append(high_freq_energy / total_energy)
        
        # Return average high-frequency energy ratio
        return np.mean(dct_scores) if dct_scores else 0.0
    
    def _analyze_wavelets(self, image):
        """Analyze wavelet coefficients."""
        # Multi-level wavelet decomposition
        coeffs = pywt.wavedec2(image, self.wavelet_type, level=3)
        
        # Analyze detail coefficients
        detail_energies = []
        
        for level_coeffs in coeffs[1:]:  # Skip approximation
            for detail in level_coeffs:  # cH, cV, cD
                energy = np.sum(np.abs(detail) ** 2)
                detail_energies.append(energy)
        
        # High detail energy might indicate adversarial perturbations
        total_energy = np.sum(image.astype(np.float32) ** 2)
        detail_ratio = sum(detail_energies) / (total_energy + 1e-8)
        
        return min(detail_ratio * 1000, 1.0)  # Scale and cap at 1.0
    
    def _analyze_fft(self, image):
        """Analyze FFT spectrum."""
        # Compute 2D FFT
        fft_image = fft2(image)
        magnitude_spectrum = np.abs(fftshift(fft_image))
        
        # Analyze high-frequency components
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        
        # Create masks for different frequency bands
        low_freq_mask = self._create_circular_mask(h, w, center_y, center_x, min(h, w) // 6)
        high_freq_mask = ~self._create_circular_mask(h, w, center_y, center_x, min(h, w) // 3)
        
        # Calculate energy in different bands
        low_freq_energy = np.sum(magnitude_spectrum * low_freq_mask)
        high_freq_energy = np.sum(magnitude_spectrum * high_freq_mask)
        
        # High-frequency to low-frequency ratio
        if low_freq_energy > 0:
            freq_ratio = high_freq_energy / low_freq_energy
            return min(freq_ratio / 10.0, 1.0)  # Normalize
        
        return 0.0
    
    def _create_circular_mask(self, h, w, center_y, center_x, radius):
        """Create circular mask for frequency analysis."""
        y, x = np.ogrid[:h, :w]
        mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
        return mask