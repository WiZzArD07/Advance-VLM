import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import os
from pathlib import Path

class DetectionVisualizer:
    def __init__(self, config):
        self.config = config
        self.save_plots = config.get('visualization', {}).get('save_plots', True)
        self.plot_dir = config.get('visualization', {}).get('plot_dir', 'results/plots/')
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_detection_report(self, image_path, results, output_dir):
        """Create comprehensive detection report with visualizations."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Load original image
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Original image
        plt.subplot(3, 4, 1)
        if image is not None:
            plt.imshow(image)
        plt.title('Original Image', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # 2. Detection scores bar chart
        plt.subplot(3, 4, 2)
        methods = list(results['detailed_results'].keys())
        scores = [results['detailed_results'][method]['score'] for method in methods]
        
        colors = ['red' if score > 0.5 else 'green' for score in scores]
        bars = plt.bar(range(len(methods)), scores, color=colors, alpha=0.7)
        plt.axhline(y=0.5, color='orange', linestyle='--', label='Threshold')
        plt.xticks(range(len(methods)), [m.replace('_', ' ').title() for m in methods], rotation=45)
        plt.ylabel('Detection Score')
        plt.title('Detection Scores by Method', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add score labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Overall risk assessment
        plt.subplot(3, 4, 3)
        overall_score = results['overall_score']
        risk_level = 'HIGH' if overall_score > 0.7 else 'MEDIUM' if overall_score > 0.4 else 'LOW'
        risk_color = 'red' if risk_level == 'HIGH' else 'orange' if risk_level == 'MEDIUM' else 'green'
        
        plt.pie([overall_score, 1-overall_score], labels=['Adversarial', 'Clean'], 
                colors=[risk_color, 'lightgray'], autopct='%1.1f%%', startangle=90)
        plt.title(f'Risk Level: {risk_level}\nScore: {overall_score:.3f}', 
                 fontsize=14, fontweight='bold')
        
        # 4. Frequency analysis visualization (if available)
        if image is not None:
            plt.subplot(3, 4, 4)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            fft = np.fft.fft2(gray)
            magnitude_spectrum = np.log(np.abs(np.fft.fftshift(fft)) + 1)
            plt.imshow(magnitude_spectrum, cmap='hot')
            plt.title('Frequency Domain Analysis', fontsize=14, fontweight='bold')
            plt.colorbar()
            plt.axis('off')
        
        # 5-8. Individual method visualizations
        if image is not None:
            self._visualize_pixel_analysis(plt.subplot(3, 4, 5), image, results)
            self._visualize_gradient_analysis(plt.subplot(3, 4, 6), image)
            self._visualize_texture_analysis(plt.subplot(3, 4, 7), image)
            self._visualize_edge_analysis(plt.subplot(3, 4, 8), image)
        
        # 9. Detection timeline/summary
        plt.subplot(3, 4, 9)
        detection_summary = []
        for method, result in results['detailed_results'].items():
            status = 'DETECTED' if result['detected'] else 'CLEAN'
            detection_summary.append(f"{method.replace('_', ' ').title()}: {status}")
        
        plt.text(0.1, 0.5, '\n'.join(detection_summary), fontsize=10, 
                verticalalignment='center', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        plt.title('Detection Summary', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # 10. Hidden content analysis
        plt.subplot(3, 4, 10)
        if results['hidden_content']:
            hidden_text = '\n'.join(results['hidden_content'][:5])  # Limit to 5 items
            plt.text(0.1, 0.5, f"Hidden Content Found:\n\n{hidden_text}", 
                    fontsize=9, verticalalignment='center', 
                    transform=plt.gca().transAxes, wrap=True,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
        else:
            plt.text(0.1, 0.5, "No Hidden Content Detected", 
                    fontsize=12, verticalalignment='center', 
                    transform=plt.gca().transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        plt.title('Steganography Analysis', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # 11. Recommendations
        plt.subplot(3, 4, 11)
        recommendations = self._generate_recommendations(results)
        plt.text(0.1, 0.5, recommendations, fontsize=10, 
                verticalalignment='center', transform=plt.gca().transAxes, wrap=True,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.7))
        plt.title('Recommendations', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # 12. Technical details
        plt.subplot(3, 4, 12)
        tech_details = f"""Image Shape: {results['image_shape']}
File: {os.path.basename(image_path)}
Analysis Methods: {len(results['detailed_results'])}
Processing Time: < 1 second
Confidence: {'High' if overall_score > 0.8 or overall_score < 0.2 else 'Medium'}"""
        
        plt.text(0.1, 0.5, tech_details, fontsize=10, 
                verticalalignment='center', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lavender", alpha=0.7))
        plt.title('Technical Details', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save the report
        if self.save_plots:
            report_path = os.path.join(output_dir, 'detection_report.png')
            plt.savefig(report_path, dpi=300, bbox_inches='tight')
            print(f"Visualization report saved: {report_path}")
        
        plt.show()
        
        # Generate text report
        self._generate_text_report(results, output_dir, image_path)
    
    def _visualize_pixel_analysis(self, ax, image, results):
        """Visualize pixel-level anomalies."""
        plt.sca(ax)
        
        # Create a heatmap of pixel intensities
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        plt.imshow(gray, cmap='viridis')
        plt.title('Pixel Intensity Analysis', fontsize=12, fontweight='bold')
        plt.colorbar()
        plt.axis('off')
    
    def _visualize_gradient_analysis(self, ax, image):
        """Visualize gradient analysis."""
        plt.sca(ax)
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        plt.imshow(grad_magnitude, cmap='hot')
        plt.title('Gradient Magnitude', fontsize=12, fontweight='bold')
        plt.colorbar()
        plt.axis('off')
    
    def _visualize_texture_analysis(self, ax, image):
        """Visualize texture analysis."""
        plt.sca(ax)
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Local standard deviation as texture measure
        kernel = np.ones((5, 5)) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        
        plt.imshow(np.sqrt(local_variance), cmap='plasma')
        plt.title('Texture Variation', fontsize=12, fontweight='bold')
        plt.colorbar()
        plt.axis('off')
    
    def _visualize_edge_analysis(self, ax, image):
        """Visualize edge analysis."""
        plt.sca(ax)
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        plt.imshow(edges, cmap='gray')
        plt.title('Edge Detection', fontsize=12, fontweight='bold')
        plt.axis('off')
    
    def _generate_recommendations(self, results):
        """Generate recommendations based on detection results."""
        overall_score = results['overall_score']
        
        if overall_score > 0.7:
            return """HIGH RISK DETECTED:
• Block this image immediately
• Investigate source and intent
• Check for policy violations
• Consider forensic analysis"""
        elif overall_score > 0.4:
            return """MEDIUM RISK:
• Review image manually
• Check context of usage
• Monitor for patterns
• Apply additional filters"""
        else:
            return """LOW RISK:
• Image appears clean
• Normal processing allowed
• Continue monitoring
• No immediate action needed"""
    
    def _generate_text_report(self, results, output_dir, image_path):
        """Generate detailed text report."""
        report_path = os.path.join(output_dir, 'detection_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("ADVERSARIAL IMAGE DETECTION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Image: {os.path.basename(image_path)}\n")
            f.write(f"Analysis Date: {plt.datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Image Shape: {results['image_shape']}\n\n")
            
            f.write(f"OVERALL ASSESSMENT:\n")
            f.write(f"Score: {results['overall_score']:.4f}\n")
            f.write(f"Classification: {'ADVERSARIAL' if results['is_adversarial'] else 'CLEAN'}\n\n")
            
            f.write("DETAILED ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            
            for method, result in results['detailed_results'].items():
                f.write(f"\n{method.replace('_', ' ').title()}:\n")
                f.write(f"  Score: {result['score']:.4f}\n")
                f.write(f"  Status: {'DETECTED' if result['detected'] else 'CLEAN'}\n")
                
                # Add method-specific details
                if 'statistics' in result:
                    f.write(f"  Additional Details: Statistical analysis completed\n")
                if 'anomaly_score' in result:
                    f.write(f"  Anomaly Score: {result['anomaly_score']:.4f}\n")
            
            if results['hidden_content']:
                f.write(f"\nHIDDEN CONTENT DETECTED:\n")
                f.write("-" * 30 + "\n")
                for content in results['hidden_content']:
                    f.write(f"• {content}\n")
            
            f.write(f"\nRECOMMENDATIONS:\n")
            f.write("-" * 30 + "\n")
            f.write(self._generate_recommendations(results))
        
        print(f"Text report saved: {report_path}")