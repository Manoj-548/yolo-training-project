import os
import json
import yaml
from pathlib import Path
from collections import defaultdict

def analyze_balanced_dataset():
    base_path = Path(__file__).parent
    balanced_dir = base_path / "balanced_dataset"
    
    # Load class configuration
    with open(base_path / "project_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Load early stopping configuration
    with open(balanced_dir / "early_stopping.json", 'r') as f:
        early_stop_config = json.load(f)
    
    # Initialize statistics
    stats = {
        'total_images': 0,
        'total_labels': 0,
        'class_distribution': defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0}),
        'split_totals': {'train': 0, 'val': 0, 'test': 0}
    }
    
    # Analyze images and labels
    for split in ['train', 'val', 'test']:
        img_dir = balanced_dir / "images" / split
        label_dir = balanced_dir / "labels" / split
        
        if not img_dir.exists():
            continue
            
        # Count images per class
        for img_path in img_dir.glob("*.*"):
            stats['total_images'] += 1
            stats['split_totals'][split] += 1
            
            # Extract class name from filename
            class_name = img_path.stem.split('_')[0]
            # Handle mapped special characters
            for char, mapped in {
                'DQUOTE': '"', 'SQUOTE': "'", 'ASTERISK': '*',
                'PLUS': '+', 'COMMA': ',', 'HYPHEN': '-',
                'DOT': '.', 'FSLASH': '/', 'COLON': ':',
                'SEMICOLON': ';', 'LT': '<', 'EQUALS': '=',
                'GT': '>', 'QMARK': '?', 'LBRACKET': '[',
                'BSLASH': '\\', 'RBRACKET': ']', 'CARET': '^',
                'UNDERSCORE': '_', 'BACKTICK': '`', 'LCURLY': '{',
                'PIPE': '|', 'RCURLY': '}', 'TILDE': '~'
            }.items():
                if class_name == mapped:
                    class_name = char
                    break
            
            stats['class_distribution'][class_name][split] += 1
    
    # Print analysis
    print("\n=== Balanced Dataset Analysis ===")
    print(f"\nTotal Images: {stats['total_images']}")
    print("\nSplit Distribution:")
    for split, count in stats['split_totals'].items():
        percentage = (count / stats['total_images']) * 100
        print(f"{split}: {count} images ({percentage:.1f}%)")
    
    print("\nClass Distribution:")
    print("Class\t\tTrain\tVal\tTest\tTotal\tTarget Metrics")
    print("-" * 80)
    
    for class_name in sorted(stats['class_distribution'].keys()):
        dist = stats['class_distribution'][class_name]
        total = sum(dist.values())
        
        # Get target metrics for this class
        targets = early_stop_config['class_targets'].get(class_name, {})
        metrics = f"P:{targets.get('precision', 0):.2f} R:{targets.get('recall', 0):.2f} F1:{targets.get('f1', 0):.2f}"
        
        # Handle special character display
        display_name = class_name if len(class_name) == 1 else f"'{class_name}'"
        print(f"{display_name:<8}\t{dist['train']}\t{dist['val']}\t{dist['test']}\t{total}\t{metrics}")
    
    print("\nEarly Stopping Configuration:")
    print(f"Minimum Epochs: {early_stop_config.get('min_epochs', 'N/A')}")
    print(f"Maximum Epochs: {early_stop_config.get('max_epochs', 'N/A')}")
    print(f"Evaluation Frequency: Every {early_stop_config.get('evaluation_frequency', 'N/A')} epochs")
    print(f"Patience: {early_stop_config.get('patience', 'N/A')} evaluations")

if __name__ == "__main__":
    analyze_balanced_dataset()