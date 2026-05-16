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
        img = np.random.normal(128, 10, (224, 224, 3)).astype(np.uint8)
        cv2.imwrite(str(roots[0] / f"good_{i:03d}.jpg"), img)

    # Generate 10 Test Good
    for i in range(10):
        img = np.random.normal(128, 10, (224, 224, 3)).astype(np.uint8)
        cv2.imwrite(str(roots[1] / f"test_good_{i:03d}.jpg"), img)

    # Generate 10 Test Bad (add a black spot)
    for i in range(10):
        img = np.random.normal(128, 10, (224, 224, 3)).astype(np.uint8)
        # Defect
        cv2.circle(img, (112, 112), 20, (0, 0, 0), -1)
        cv2.imwrite(str(roots[2] / f"test_bad_{i:03d}.jpg"), img)
        
    print("[SUCCESS] Synthetic data generated.")

if __name__ == "__main__":
    generate_synthetic_data()
