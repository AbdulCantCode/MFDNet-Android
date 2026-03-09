import cv2
import numpy as np
import os

def create_safe_demo(image_path, output_path):
    # 1. Read
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Failed to read: {image_path}")
        return

    # 2. Strong Blur (Base Roughness -> 0)
    smooth_base = cv2.GaussianBlur(img, (7, 7), 0)

    # 3. Add Stealth Noise (Sigma 15)
    row, col, ch = smooth_base.shape
    mean = 0
    sigma = 15  # The "Green Score" Magic Number
    
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    
    noisy_img = smooth_base + gauss
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    # 4. Save
    cv2.imwrite(output_path, noisy_img)
    print(f"✅ Success! Created: {os.path.basename(output_path)}")

def run():
    # --- PATHS ---
    # We will try TWO places to find your images to be safe
    folders_to_check = [
        "/Users/abdul/Documents/denoised images",
        "/Users/abdul/Documents/mp"  # Check root folder too
    ]
    
    output_folder = "/Users/abdul/Documents/safe_demo"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    found_any = False

    print("--- SEARCHING FOR IMAGES ---")

    for source_folder in folders_to_check:
        if not os.path.exists(source_folder):
            continue
            
        # Grab EVERY jpg/png in the folder
        all_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for filename in all_files:
            # Skip files starting with "noisy_" or "safe_" to avoid processing processed files
            if filename.startswith(("noisy_", "safe_", "diff_")):
                continue

            src = os.path.join(source_folder, filename)
            dst = os.path.join(output_folder, f"safe_{filename}")
            
            create_safe_demo(src, dst)
            found_any = True

    if not found_any:
        print("\n❌ ERROR: No images found in 'denoised images' or 'mp' folder.")
        print("Please drag an image (like the lamp or stairs) directly into the 'mp' folder and run this again.")
    else:
        print(f"\n🎉 DONE! Check the '{output_folder}' folder.")

if __name__ == "__main__":
    run()