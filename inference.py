import os
import torch
import numpy as np
import cv2
from mfdnet import MFDNet

# --- CONFIGURATION ---
PROJECT_PATH = "." 
INPUT_FOLDER = "test_images"  
OUTPUT_FOLDER = "denoised_output"
MODEL_PATH = "best_model.pth"

# --- SETUP DEVICE & PRECISION ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using Apple M1 GPU (MPS)")
    use_fp16 = True  # M1 is great at FP16
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀 Using NVIDIA GPU")
    use_fp16 = True
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU (FP16 disabled)")
    use_fp16 = False

def load_image(path):
    img = cv2.imread(path)
    if img is None: return None, None
    
    h, w = img.shape[:2]
    
    # Convert to RGB and Normalize
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    
    # To Tensor
    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device)
    
    # OPTIMIZATION: Convert to Half Precision (FP16)
    if use_fp16:
        tensor = tensor.half()
        
    return tensor, (h, w)

def save_image(tensor, path):
    # Convert back to float32 for saving (libraries prefer 32-bit for I/O)
    img = tensor.float().squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)

def main():
    # 1. Load Model
    model = MFDNet(in_ch=3, base_ch=32, num_blocks=4).to(device)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        
        # OPTIMIZATION: Cast model to FP16
        if use_fp16:
            model.half()
            print("⚡ Optimized: Model converted to FP16 (Half-Precision)")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Prepare Folders
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"⚠️ Created folder '{INPUT_FOLDER}'. Please put some noisy images there!")
        return

    # 3. Run on all images
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images. Processing...")

    for img_name in image_files:
        in_path = os.path.join(INPUT_FOLDER, img_name)
        out_path = os.path.join(OUTPUT_FOLDER, "denoised_" + img_name)
        
        img_t, _ = load_image(in_path)
        if img_t is None: continue

        # OPTIMIZATION: inference_mode is faster than no_grad
        with torch.inference_mode():
            output_t = model(img_t)

        save_image(output_t, out_path)
        print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    main()