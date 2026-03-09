import cv2
import numpy as np
import os

def create_uniform_demo(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        return

    # 1. BLUR: Kill texture so the base is smooth
    smooth_base = cv2.GaussianBlur(img, (7, 7), 0)

    # 2. UNIFORM NOISE (The Fix)
    # Instead of a "bell curve" (Gaussian), we use a "box".
    # We set the range to [-24, 24].
    # This guarantees NO pixel difference is >= 25.
    # Your app will count 100% of this noise.
    row, col, ch = smooth_base.shape
    
    # Generate random values strictly between -24 and 24
    noise = np.random.uniform(-24, 24, (row, col, ch))
    
    # Add noise
    noisy_img = smooth_base + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    cv2.imwrite(output_path, noisy_img)
    print(f"✅ Processed (Uniform 24): {os.path.basename(image_path)}")

def run():
    # --- CONFIGURATION ---
    # Check ALL folders to find your phone photo
    folders_to_check = [
        "/Users/abdul/Documents/noisy images",
        "/Users/abdul/Documents/denoised images",
        "/Users/abdul/Documents/mp"
    ]
    
    output_folder = "/Users/abdul/Documents/final_demo"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("--- CREATING FINAL DEMO IMAGES (Uniform Noise) ---")
    
    processed_count = 0
    for folder in folders_to_check:
        if not os.path.exists(folder): continue
        
        # Get all images
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for f in files:
            # Skip files we already made
            if f.startswith(("noisy_", "safe_", "demo_", "diff_", "final_")): continue
            
            src = os.path.join(folder, f)
            dst = os.path.join(output_folder, f"final_{f}")
            
            create_uniform_demo(src, dst)
            processed_count += 1

    if processed_count > 0:
        print(f"\n🎉 Success! {processed_count} images saved to 'final_demo'.")
        print("Transfer these 'final_' images to your phone.")
    else:
        print("\n❌ No images found! Please drag your phone photo into the 'mp' folder.")

if __name__ == "__main__":
    run()