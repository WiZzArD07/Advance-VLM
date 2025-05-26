import os
import sys
import argparse
import yaml
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from detector import AdversarialDetector
from utils.visualization import DetectionVisualizer

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Advanced Adversarial Image Detection")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--output", type=str, default="results/", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize detector
    detector = AdversarialDetector(config)
    visualizer = DetectionVisualizer(config)
    
    # Analyze image
    print(f"Analyzing image: {args.image}")
    results = detector.detect_adversarial(args.image, verbose=args.verbose)
    
    # Display results
    print("\n" + "="*60)
    print("ADVERSARIAL DETECTION RESULTS")
    print("="*60)
    
    print(f"Overall Adversarial Score: {results['overall_score']:.3f}")
    print(f"Classification: {'ADVERSARIAL' if results['is_adversarial'] else 'CLEAN'}")
    
    print("\nDetailed Analysis:")
    for method, result in results['detailed_results'].items():
        print(f"  {method}: {result['score']:.3f} ({'DETECTED' if result['detected'] else 'CLEAN'})")
    
    if results['hidden_content']:
        print("\nPotential Hidden Content Detected:")
        for content in results['hidden_content']:
            print(f"  - {content}")
    
    # Generate visualization
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        visualizer.create_detection_report(args.image, results, args.output)
        print(f"\nDetailed report saved to: {args.output}")

if __name__ == "__main__":
    main()