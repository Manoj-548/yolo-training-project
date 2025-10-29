import pandas as pd
import matplotlib.pyplot as plt
import psutil
import os
from pathlib import Path
import time
import numpy as np
import torch
import seaborn as sns
from datetime import datetime
import json

class TrainingDashboard:
    def __init__(self):
        self.results_path = Path("runs/train/results.csv")
        self.model_path = Path("runs/train/weights/best.pt")
        self.start_time = datetime.now()
        
        # Initialize plots
        plt.ion()
        self.fig = plt.figure(figsize=(15, 10))
        self.gs = self.fig.add_gridspec(3, 3)
        self.setup_plots()
        
        # Initialize metrics storage
        self.cpu_history = []
        self.memory_history = []
        self.gpu_history = []
        self.time_history = []
        
        # Save metrics to file for persistence
        self.metrics_file = Path("training_metrics.json")
        self.load_saved_metrics()
        
    def setup_plots(self):
        """Initialize all subplot axes"""
        # Training metrics
        self.ax_loss = self.fig.add_subplot(self.gs[0, :2])
        self.ax_map = self.fig.add_subplot(self.gs[1, :2])
        
        # Resource usage
        self.ax_cpu = self.fig.add_subplot(self.gs[0, 2])
        self.ax_memory = self.fig.add_subplot(self.gs[1, 2])
        self.ax_gpu = self.fig.add_subplot(self.gs[2, 2])
        
        # Dataset info and model performance
        self.ax_dataset = self.fig.add_subplot(self.gs[2, 0])
        self.ax_model = self.fig.add_subplot(self.gs[2, 1])
        
        self.fig.suptitle('Enhanced Training Performance Dashboard')
        plt.tight_layout()

    def get_system_metrics(self):
        """Collect system resource usage metrics"""
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        # GPU metrics if available
        gpu_percent = 0
        if torch.cuda.is_available():
            gpu_percent = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100 if torch.cuda.max_memory_allocated() > 0 else 0
            
        return cpu_percent, memory_percent, gpu_percent

    def update_resource_plots(self):
        """Update system resource usage plots"""
        cpu, memory, gpu = self.get_system_metrics()
        current_time = (datetime.now() - self.start_time).total_seconds() / 60  # Minutes
        
        self.cpu_history.append(cpu)
        self.memory_history.append(memory)
        self.gpu_history.append(gpu)
        self.time_history.append(current_time)
        
        # Update plots
        self._plot_resource(self.ax_cpu, self.time_history, self.cpu_history, 'CPU Usage (%)')
        self._plot_resource(self.ax_memory, self.time_history, self.memory_history, 'Memory Usage (%)')
        self._plot_resource(self.ax_gpu, self.time_history, self.gpu_history, 'GPU Memory Usage (%)')

    def _plot_resource(self, ax, x, y, title):
        """Helper function for resource plotting"""
        ax.clear()
        ax.plot(x, y)
        ax.set_title(title)
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Usage %')
        ax.grid(True)

    def update_training_metrics(self):
        """Update training metrics from results.csv"""
        if self.results_path.exists():
            df = pd.read_csv(self.results_path)
            # Clean column names by stripping whitespace
            df.columns = df.columns.str.strip()
            
            # Plot losses
            self.ax_loss.clear()
            self.ax_loss.plot(df['epoch'], df['train/box_loss'], label='Box Loss')
            self.ax_loss.plot(df['epoch'], df['train/cls_loss'], label='Class Loss')
            self.ax_loss.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss')
            self.ax_loss.set_title('Training Losses')
            self.ax_loss.legend()
            self.ax_loss.grid(True)
            
            # Plot mAP
            self.ax_map.clear()
            self.ax_map.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
            self.ax_map.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
            self.ax_map.set_title('Model Performance (mAP)')
            self.ax_map.legend()
            self.ax_map.grid(True)

    def update_dataset_info(self):
        """Update dataset statistics visualization"""
        # Count images in each split
        train_images = len(list(Path('images/train').glob('*.jpg')))
        val_images = len(list(Path('images/val').glob('*.jpg')))
        test_images = len(list(Path('images/test').glob('*.jpg')))
        
        self.ax_dataset.clear()
        splits = ['Train', 'Validation', 'Test']
        counts = [train_images, val_images, test_images]
        
        bars = self.ax_dataset.bar(splits, counts)
        self.ax_dataset.set_title('Dataset Distribution')
        self.ax_dataset.set_ylabel('Number of Images')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            self.ax_dataset.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom')

    def update_model_metrics(self):
        """Update model performance metrics"""
        if self.results_path.exists():
            df = pd.read_csv(self.results_path)
            # Clean column names by stripping whitespace
            df.columns = df.columns.str.strip()
            if not df.empty:
                latest_metrics = df.iloc[-1]
                
                metrics_text = f"""Latest Metrics:
                mAP50: {latest_metrics['metrics/mAP50(B)']:.3f}
                mAP50-95: {latest_metrics['metrics/mAP50-95(B)']:.3f}
                Precision: {latest_metrics['metrics/precision(B)']:.3f}
                Recall: {latest_metrics['metrics/recall(B)']:.3f}
                Epoch: {latest_metrics['epoch']}"""
                
                self.ax_model.clear()
                self.ax_model.text(0.1, 0.5, metrics_text, fontsize=10, va='center')
                self.ax_model.set_title('Current Model Metrics')
                self.ax_model.axis('off')

    def load_saved_metrics(self):
        """Load previously saved metrics if they exist"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.cpu_history = data.get('cpu', [])
                    self.memory_history = data.get('memory', [])
                    self.gpu_history = data.get('gpu', [])
                    self.time_history = data.get('time', [])
            except Exception as e:
                print(f"Error loading saved metrics: {e}")

    def save_metrics(self):
        """Save current metrics to file"""
        try:
            data = {
                'cpu': self.cpu_history,
                'memory': self.memory_history,
                'gpu': self.gpu_history,
                'time': self.time_history
            }
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving metrics: {e}")

    def update(self):
        """Update all dashboard components"""
        try:
            self.update_resource_plots()
            self.update_training_metrics()
            self.update_dataset_info()
            self.update_model_metrics()
            
            plt.tight_layout()
            plt.draw()
            plt.pause(1)  # Update every second
            
            # Save metrics periodically
            self.save_metrics()
            
        except Exception as e:
            print(f"Error updating dashboard: {e}")

if __name__ == "__main__":
    # Create and run dashboard
    dashboard = TrainingDashboard()
    
    try:
        while True:
            dashboard.update()
            time.sleep(1)  # Update interval
            
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Dashboard error: {e}")
    finally:
        plt.close('all')