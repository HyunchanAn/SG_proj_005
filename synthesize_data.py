import numpy as np
import cv2
import os
from pathlib import Path

def generate_synthetic_data():
    roots = [
        Path("datasets/custom/train/good"),
        Path("datasets/custom/test/good"),
        Path("datasets/custom/test/bad")
    ]
    
    for r in roots:
        r.mkdir(parents=True, exist_ok=True)

    print("[INFO] Generating synthetic data...")

    # Generate 50 Train Good
    for i in range(50):
        # Gray background + noise
        img = np.random.normal(128, 10, (512, 512, 3)).astype(np.uint8)
        cv2.imwrite(str(roots[0] / f"good_{i:03d}.jpg"), img)

    # Generate 10 Test Good
    for i in range(10):
        img = np.random.normal(128, 10, (512, 512, 3)).astype(np.uint8)
        cv2.imwrite(str(roots[1] / f"test_good_{i:03d}.jpg"), img)

    # Generate 10 Test Bad (add a black spot)
    for i in range(10):
        img = np.random.normal(128, 10, (512, 512, 3)).astype(np.uint8)
        # Defect - adjusted for 512x512
        cv2.circle(img, (256, 256), 40, (0, 0, 0), -1)
        cv2.imwrite(str(roots[2] / f"test_bad_{i:03d}.jpg"), img)
        
    print("[SUCCESS] Synthetic data generated.")

if __name__ == "__main__":
    generate_synthetic_data()
