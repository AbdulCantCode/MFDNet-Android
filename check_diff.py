import cv2
import numpy as np

# Load the images
noisy = cv2.imread("test_images/test_plant.jpg")
denoised = cv2.imread("denoised_output/denoised_test_plant.jpg")

if noisy is None or denoised is None:
    print("❌ Check filenames. Run inference.py first!")
else:
    # Calculate absolute difference
    diff = cv2.absdiff(noisy, denoised)

    # Multiply by 10 so your eyes can see the faint noise
    diff_amplified = diff * 10

    cv2.imwrite("diff_map.png", diff_amplified)
    print("✅ Created 'diff_map.png'. Open it!")