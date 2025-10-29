import os
import yaml
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

class DatasetAnalyzer:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.config_file = self.base_path / "project_config.yaml"
        self.stats_file = self.base_path / "dataset_stats.json"
        
        # Load configurations
        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load current stats
        with open(self.stats_file, 'r') as f:
            self.stats = json.load(f)
            
    def analyze_class_balance(self):
        """Analyze class distribution and identify training needs"""
        distributions = self.stats['class_distribution']
        counts = np.array(list(distributions.values()))
        non_zero_counts = counts[counts > 0]
        
        # Calculate statistics
        mean_samples = np.mean(non_zero_counts)
        median_samples = np.median(non_zero_counts)
        std_samples = np.std(non_zero_counts)
        
        # Define thresholds
        well_trained_threshold = mean_samples + std_samples
        undertrained_threshold = mean_samples - std_samples
        
        # Categorize classes
        class_status = {
            'well_trained': [],
            'adequately_trained': [],
            'undertrained': [],
            'missing': []
        }
        
        for class_name, count in distributions.items():
            if count == 0:
                class_status['missing'].append((class_name, count))
            elif count > well_trained_threshold:
                class_status['well_trained'].append((class_name, count))
            elif count < undertrained_threshold:
                class_status['undertrained'].append((class_name, count))
            else:
                class_status['adequately_trained'].append((class_name, count))
        
        # Calculate training recommendations
        target_samples = int(median_samples)  # Use median as target sample count
        recommendations = {
            'target_samples': target_samples,
            'synthetic_needed': {name: target_samples for name, _ in class_status['missing']},
            'reduction_needed': {name: count - target_samples 
                               for name, count in class_status['well_trained']
                               if count > target_samples},
            'augmentation_needed': {name: target_samples - count 
                                  for name, count in class_status['undertrained']
                                  if count < target_samples}
        }
        
        return {
            'statistics': {
                'mean_samples': mean_samples,
                'median_samples': median_samples,
                'std_samples': std_samples,
                'well_trained_threshold': well_trained_threshold,
                'undertrained_threshold': undertrained_threshold
            },
            'class_status': class_status,
            'recommendations': recommendations
        }
    
    def generate_training_config(self):
        """Generate focused training configuration based on analysis"""
        analysis = self.analyze_class_balance()
        
        # Calculate epochs based on training needs
        total_undertrained = len(analysis['class_status']['undertrained'])
        total_missing = len(analysis['class_status']['missing'])
        
        # More epochs needed if more classes need training
        base_epochs = 25  # Base number of epochs
        epoch_multiplier = (total_undertrained + total_missing) / 92  # Adjust based on proportion of classes needing training
        recommended_epochs = int(base_epochs * (1 + epoch_multiplier))
        
        training_config = {
            'focus_classes': {
                'priority_training': [name for name, _ in analysis['class_status']['undertrained']],
                'new_classes': [name for name, _ in analysis['class_status']['missing']],
                'well_trained': [name for name, _ in analysis['class_status']['well_trained']]
            },
            'training_parameters': {
                'epochs': recommended_epochs,
                'batch_size': 16,
                'initial_learning_rate': 0.001,
                'target_samples_per_class': analysis['recommendations']['target_samples']
            }
        }
        
        return training_config
    
    def print_analysis(self):
        """Print detailed analysis and recommendations"""
        analysis = self.analyze_class_balance()
        training_config = self.generate_training_config()
        
        print("\n=== Dataset Analysis Report ===")
        print("\nClass Distribution Statistics:")
        print(f"Mean samples per class: {analysis['statistics']['mean_samples']:.2f}")
        print(f"Median samples per class: {analysis['statistics']['median_samples']:.2f}")
        print(f"Standard deviation: {analysis['statistics']['std_samples']:.2f}")
        
        print("\nWell-trained Classes (Can reduce samples):")
        for name, count in sorted(analysis['class_status']['well_trained']):
            reduction = analysis['recommendations']['reduction_needed'].get(name, 0)
            print(f"  {name}: {count} samples (can reduce by {reduction})")
        
        print("\nAdequately Trained Classes (No action needed):")
        for name, count in sorted(analysis['class_status']['adequately_trained']):
            print(f"  {name}: {count} samples")
        
        print("\nUnder-trained Classes (Need more samples):")
        for name, count in sorted(analysis['class_status']['undertrained']):
            needed = analysis['recommendations']['augmentation_needed'].get(name, 0)
            print(f"  {name}: {count} samples (need {needed} more)")
        
        print("\nMissing Classes (Need synthetic data):")
        for name, _ in sorted(analysis['class_status']['missing']):
            needed = analysis['recommendations']['synthetic_needed'].get(name, 0)
            print(f"  {name}: need {needed} samples")
        
        print("\n=== Training Recommendations ===")
        print(f"\nRecommended Epochs: {training_config['training_parameters']['epochs']}")
        print(f"Target Samples per Class: {training_config['training_parameters']['target_samples_per_class']}")
        print("\nPriority Training Classes:")
        for class_name in training_config['focus_classes']['priority_training']:
            print(f"  {class_name}")
        
        # Save training configuration
        config_path = self.base_path / "training_focus.json"
        with open(config_path, 'w') as f:
            json.dump(training_config, f, indent=2)
        print(f"\nTraining configuration saved to: {config_path}")

if __name__ == "__main__":
    analyzer = DatasetAnalyzer()
    analyzer.print_analysis()