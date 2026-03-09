import os
import cv2
import torch
import numpy as np
from mfdnet import MFDNet  # Validates your model architecture is present

def run_full_pipeline():
    # --- PATHS ---
    # We use absolute paths to avoid confusion
    noisy_folder = "/Users/abdul/Documents/noisy images"
    denoised_folder = "/Users/abdul/Documents/denoised images"
    diff_folder = "/Users/abdul/Documents/difference maps"
    model_path = "best_model.pth"

    # Create folders if they don't exist
    for folder in [denoised_folder, diff_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # --- 1. SETUP MODEL ---
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"🚀 Pipeline started on device: {device}")

    model = MFDNet(in_ch=3, base_ch=32, num_blocks=4).to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Model loaded.")
    else:
        print(f"❌ Error: '{model_path}' not found. Please put it in the 'mp' folder.")
        return

    model.eval()

    # --- 2. PROCESS LOOP (Denoise -> Save -> Diff Map) ---
    files = [f for f in os.listdir(noisy_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    print(f"Found {len(files)} images to process.\n")

    with torch.no_grad():
        for filename in files:
            # Paths
            noisy_path = os.path.join(noisy_folder, filename)
            denoised_path = os.path.join(denoised_folder, filename)
            diff_path = os.path.join(diff_folder, f"diff_{filename}")

            # A. READ & DENOISE
            img_bgr = cv2.imread(noisy_path)
            if img_bgr is None:
                print(f"⚠️ Failed to read: {filename}")
                continue
            
            # Preprocess
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_float = img_rgb.astype(np.float32) / 255.0
            input_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # Inference
            output_tensor = model(input_tensor)
            
            # Postprocess
            output_tensor = output_tensor.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1)
            output_img = (output_tensor.numpy() * 255.0).astype(np.uint8)
            output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

            # B. SAVE DENOISED IMAGE
            success = cv2.imwrite(denoised_path, output_bgr)
            if not success:
                print(f"❌ Failed to write denoised image: {filename}")
                continue
            
            # C. GENERATE DIFFERENCE MAP
            # Load the noisy image as float again for math
            noisy_float = img_bgr.astype(np.float32)
            clean_float = output_bgr.astype(np.float32)

            # Calculate Difference
            diff = np.abs(noisy_float - clean_float)
            
            # Boost brightness (x15) to make noise visible
            diff_boosted = diff * 15.0
            diff_boosted = np.clip(diff_boosted, 0, 255).astype(np.uint8)

            # Save Diff Map
            cv2.imwrite(diff_path, diff_boosted)
            print(f"✅ Processed: {filename} -> Saved Denoised & Diff Map")

    print(f"\n🎉 All done! Check this folder: {diff_folder}")

if __name__ == "__main__":
    run_full_pipeline()