import os
from PIL import Image

DATA_DIR = "data"

def verify_images():
    images = [f for f in os.listdir(DATA_DIR) if f.endswith(".jpg")]
    print(f"Found {len(images)} images in {DATA_DIR}")
    
    for img_name in images:
        path = os.path.join(DATA_DIR, img_name)
        try:
            with Image.open(path) as img:
                print(f"  - {img_name}: {img.format}, {img.size}, {img.mode}")
        except Exception as e:
            print(f"  - {img_name}: FAILED to load - {e}")

if __name__ == "__main__":
    verify_images()
