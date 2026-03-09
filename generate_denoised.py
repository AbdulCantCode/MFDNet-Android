import os
import cv2
import torch
import numpy as np
from mfdnet import MFDNet  # Must be in the same folder

def save_denoised_images():
    # --- CONFIGURATION ---
    noisy_folder = "/Users/abdul/Documents/noisy images"
    output_folder = "/Users/abdul/Documents/denoised images"
    model_path = "best_model.pth"

    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # --- 1. LOAD MODEL ---
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"🚀 Processing on: {device}")

    # Standard MFDNet Configuration (32 channels, 4 blocks)
    model = MFDNet(in_ch=3, base_ch=32, num_blocks=4).to(device)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Model weights loaded.")
    else:
        print(f"❌ Error: '{model_path}' not found.")
        return

    model.eval()

    # --- 2. PROCESS IMAGES ---
    files = [f for f in os.listdir(noisy_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(files)} images to process...")

    with torch.no_grad():
        for filename in files:
            # Paths
            input_path = os.path.join(noisy_folder, filename)
            save_path = os.path.join(output_folder, filename)

            # Read Image
            img_bgr = cv2.imread(input_path)
            if img_bgr is None:
                continue

            # Preprocess (Normalize 0-1)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_float = img_rgb.astype(np.float32) / 255.0
            input_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)

            # Inference
            output_tensor = model(input_tensor)

            # Postprocess (Denormalize 0-255)
            output_tensor = output_tensor.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1)
            output_img = (output_tensor.numpy() * 255.0).astype(np.uint8)
            output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

            # Save the Denoised Image
            cv2.imwrite(save_path, output_bgr)
            print(f"✨ Saved Denoised Image: {filename}")

    print(f"\n🎉 Done! Images saved to: {output_folder}")

if __name__ == "__main__":
    save_denoised_images()