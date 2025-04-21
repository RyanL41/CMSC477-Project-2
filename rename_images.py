import os
from pathlib import Path
import re

# Set your paths here
main_directory = r"C:\Users\rydog\CMSC477\Images"
new_directory = r"C:\Users\rydog\CMSC477\Dataset"

def get_max_frame_number(directory):
    """Find the highest frame_XXX number in the directory"""
    pattern = re.compile(r"frame_(\d+)\.jpg$", re.IGNORECASE)
    max_num = -1
    for file in os.listdir(directory):
        match = pattern.match(file)
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)
    return max_num

def rename_images(main_dir, new_dir):
    """Rename images in new_dir to continue numbering from max frame in main_dir"""
    try:
        start_num = get_max_frame_number(main_dir) + 1
    except FileNotFoundError:
        print(f"Error: Main directory '{main_dir}' not found.")
        return
    
    if not os.path.exists(new_dir):
        print(f"Error: New directory '{new_dir}' not found.")
        return

    image_extensions = {'.jpg','.png'}
    images = [
        os.path.join(new_dir, file)
        for file in os.listdir(new_dir)
        if os.path.isfile(os.path.join(new_dir, file)) and Path(file).suffix.lower() in image_extensions
    ]

    if not images:
        print(f"No image files found in '{new_dir}'")
        return
    
    images.sort()

    for i, old_path in enumerate(images, start=start_num):
        extension = Path(old_path).suffix
        new_name = f"frame_{i}{extension}"
        new_path = os.path.join(new_dir, new_name)

        while os.path.exists(new_path):
            i += 1
            new_name = f"frame_{i}{extension}"
            new_path = os.path.join(new_dir, new_name)
        
        try:
            os.rename(old_path, new_path)
            print(f"Renamed '{os.path.basename(old_path)}' to '{new_name}'")
        except OSError as e:
            print(f"Error renaming '{old_path}': {e}")

if __name__ == "__main__":
    rename_images(main_directory, new_directory)