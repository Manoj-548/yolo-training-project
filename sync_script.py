import os
import subprocess
import time
from pathlib import Path

def sync_to_cloud():
    """Sync project to GitHub and cloud drives"""
    project_dir = Path(__file__).parent

    # Git sync
    try:
        os.chdir(project_dir)
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '-m', 'Auto-sync update'], check=True)
        subprocess.run(['git', 'push'], check=True)
        print("Git sync completed")
    except subprocess.CalledProcessError as e:
        print(f"Git sync failed: {e}")

    # OneDrive sync (if available)
    onedrive_path = Path("C:/Users/Acer/OneDrive")
    if onedrive_path.exists():
        try:
            subprocess.run(['robocopy', str(project_dir), str(onedrive_path / "yolo-training-project"), '/MIR'], check=True)
            print("OneDrive sync completed")
        except subprocess.CalledProcessError as e:
            print(f"OneDrive sync failed: {e}")

    # Google Drive sync (if available)
    gdrive_path = Path("C:/Users/Acer/Google Drive")
    if gdrive_path.exists():
        try:
            subprocess.run(['robocopy', str(project_dir), str(gdrive_path / "yolo-training-project"), '/MIR'], check=True)
            print("Google Drive sync completed")
        except subprocess.CalledProcessError as e:
            print(f"Google Drive sync failed: {e}")

if __name__ == "__main__":
    while True:
        sync_to_cloud()
        time.sleep(3600)  # Sync every hour
