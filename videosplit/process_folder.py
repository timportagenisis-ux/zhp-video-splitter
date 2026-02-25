
import os
import subprocess
import sys

def process_folder(folder_path):
    """
    Processes all video files in a given folder using main.py.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}", file=sys.stderr)
        sys.exit(1)

    video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in video_extensions):
            video_path = os.path.join(folder_path, filename)
            print(f"Processing video: {video_path}")
            
            try:
                # Assuming main.py is in the same directory and executable
                subprocess.run([sys.executable, 'main.py', video_path], check=True)
                print(f"Finished processing {filename}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {filename}: {e}", file=sys.stderr)
            except FileNotFoundError:
                print("Error: main.py not found. Make sure it's in the same directory.", file=sys.stderr)
                sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_folder.py <folder_path>")
        sys.exit(1)
        
    input_folder = sys.argv[1]
    process_folder(input_folder)
