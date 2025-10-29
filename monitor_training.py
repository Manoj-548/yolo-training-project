import os
from pathlib import Path
import psutil
import json
import torch
import yaml
from datetime import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns

class TrainingMonitor:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.stats_file = self.base_path / "dataset_stats.json"
        self.progress_file = self.base_path / "training_progress.json"
        self.results_dir = self.base_path / "background_training"
        
    def check_setup(self):
        """Verify training setup and requirements"""
        checks = {
            'Dataset Stats': self.stats_file.exists(),
            'CUDA Available': torch.cuda.is_available(),
            'GPU Memory': f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB" if torch.cuda.is_available() else "N/A",
            'CPU Cores': os.cpu_count(),
            'Available RAM': f"{psutil.virtual_memory().available/1e9:.1f}GB",
            'Training Directory': self.results_dir.exists(),
            'Optimized Dataset': (self.base_path / "optimized_dataset").exists()
        }
        
        print("\nTraining Setup Check:")
        print("-" * 50)
        for check, status in checks.items():
            print(f"{check:.<30} {status}")
            
        # Load dataset stats
        if self.stats_file.exists():
            with open(self.stats_file, 'r') as f:
                stats = json.load(f)
            print("\nDataset Statistics:")
            print(f"Original Images: {stats['total_images']}")
            print(f"Optimized Images: {stats['optimized']['total_images']}")
            print(f"Classes: {len(stats['class_distribution'])}")
            
        return all(isinstance(v, bool) and v for v in checks.values())
    
    def plot_resource_usage(self):
        """Plot current system resource usage"""
        plt.figure(figsize=(12, 4))
        
        # CPU Usage
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        plt.subplot(131)
        sns.barplot(x=list(range(len(cpu_percent))), y=cpu_percent)
        plt.title('CPU Usage per Core')
        plt.xlabel('Core')
        plt.ylabel('Usage %')
        
        # Memory Usage
        memory = psutil.virtual_memory()
        plt.subplot(132)
        plt.pie([memory.used, memory.available], 
                labels=['Used', 'Available'],
                autopct='%1.1f%%')
        plt.title('Memory Usage')
        
        # GPU Usage if available
        if torch.cuda.is_available():
            plt.subplot(133)
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            total = torch.cuda.get_device_properties(0).total_memory
            plt.pie([allocated, reserved-allocated, total-reserved],
                   labels=['Allocated', 'Cached', 'Free'],
                   autopct='%1.1f%%')
            plt.title('GPU Memory Usage')
        
        plt.tight_layout()
        plt.savefig('resource_usage.png')
        plt.close()
    
    def check_training_progress(self):
        """Check current training progress"""
        if not self.progress_file.exists():
            print("\nNo training progress found.")
            return False
        
        try:
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
            
            print("\nTraining Progress:")
            print("-" * 50)
            print(f"Completed Epochs: {progress['completed_epochs']}")
            print(f"Best mAP50: {progress['best_map50']:.4f}")
            
            # Check last modification time
            last_modified = datetime.fromtimestamp(self.progress_file.stat().st_mtime)
            time_since_update = datetime.now() - last_modified
            print(f"\nLast Updated: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Time Since Update: {time_since_update}")
            
            return time_since_update.seconds < 3600  # Consider active if updated within last hour
            
        except Exception as e:
            print(f"\nError reading progress: {e}")
            return False
    
    def start_or_resume_training(self):
        """Attempt to start or resume training if not active"""
        if not self.check_training_progress():
            print("\nAttempting to start/resume training...")
            try:
                os.system(f"python background_train.py &")
                print("Training process started.")
                return True
            except Exception as e:
                print(f"Error starting training: {e}")
                return False
        return True

if __name__ == "__main__":
    monitor = TrainingMonitor()
    
    print("=== Training Monitor ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run checks
    setup_ok = monitor.check_setup()
    monitor.plot_resource_usage()
    training_active = monitor.check_training_progress()
    
    print("\nStatus Summary:")
    print("-" * 50)
    print(f"Setup Verification: {'✓' if setup_ok else '✗'}")
    print(f"Training Active: {'✓' if training_active else '✗'}")
    print(f"\nResource usage plot saved as 'resource_usage.png'")
    
    if not training_active:
        print("\nWould you like to start/resume training? (y/n)")
        if input().lower() == 'y':
            monitor.start_or_resume_training()