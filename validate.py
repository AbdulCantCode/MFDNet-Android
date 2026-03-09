import os
import torch
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from mfdnet import MFDNet
from tqdm import tqdm
import glob

# --- CONFIGURATION ---
PROJECT_PATH = "/Users/abdul/Documents/mp"
DATASET_PATH = os.path.join(PROJECT_PATH, "PolyU_Val")
MODEL_PATH = os.path.join(PROJECT_PATH, "best_model.pth")
SAVE_DEBUG_IMAGES = True

# --- DEVICE SETUP ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using M1 GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# --- UTILS ---
def load_image(path):
    """Load image, normalize 0-1, to tensor (1,3,H,W)"""
    img = cv2.imread(path)
    if img is None:
        return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device), img

def tensor_to_np(tensor):
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def calculate_tv(img):
    """
    Calculates the Total Variation (TV) metric of an image.
    
    TV is a blind quality assessment metric that measures the amount of noise 
    or high-frequency artifacts without requiring a clean ground-truth reference.
    Lower TV scores generally indicate smoother, less noisy images.
    
    Args:
        img (np.ndarray): Image scaled 0-255.
    Returns:
        float: The Total Variation score.
    """
    # Sum of absolute differences between adjacent pixels
    tv_h = np.sum(np.abs(img[:, 1:, :] - img[:, :-1, :]))
    tv_w = np.sum(np.abs(img[1:, :, :] - img[:-1, :, :]))
    return float(tv_h + tv_w) / (img.shape[0] * img.shape[1] * img.shape[2])

# --- MAIN ---
def main():
    print(f"📂 Dataset Path: {DATASET_PATH}")
    
    # 1. Load Model
    print("⏳ Loading MFDNet...")
    model = MFDNet(in_ch=3, base_ch=32, num_blocks=4).to(device)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        # Handle if weights are inside 'state_dict' key or just the dict
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Model weights loaded.")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return
    
    model.eval()

    # 2. Find Pairs (Logic: _mean.JPG is Clean, _real.JPG is Noisy)
    print("\n🔍 Pairing images...")
    # Find all clean images first
    clean_files = glob.glob(os.path.join(DATASET_PATH, "*_mean.JPG"))
    clean_files += glob.glob(os.path.join(DATASET_PATH, "*_mean.jpg")) # Case insensitive check
    
    pairs = []
    for clean_path in clean_files:
        # Construct the noisy filename by replacing _mean with _real
        if clean_path.endswith("mean.JPG"):
            noisy_path = clean_path.replace("_mean.JPG", "_real.JPG")
        else:
            noisy_path = clean_path.replace("_mean.jpg", "_real.jpg")
            
        if os.path.exists(noisy_path):
            pairs.append((noisy_path, clean_path))
    
    if len(pairs) == 0:
        print("❌ No pairs found! Check if files end with '_mean.JPG' and '_real.JPG'")
        # Debug print to help user
        print("Files in folder:", os.listdir(DATASET_PATH)[:5]) 
        return

    print(f"✅ Found {len(pairs)} pairs. Starting validation...")
    
    # 3. Run Inference
    psnr_vals = []
    ssim_vals = []
    tv_vals = []
    
    if SAVE_DEBUG_IMAGES:
        os.makedirs(os.path.join(PROJECT_PATH, "results"), exist_ok=True)

    for i, (n_path, c_path) in enumerate(tqdm(pairs)):
        # Load
        noisy_t, _ = load_image(n_path)
        clean_t, clean_np = load_image(c_path)
        
        if noisy_t is None or clean_t is None:
            continue

        # Inference (Tiling for memory safety on M1 if needed, but 512x512 might fit)
        with torch.no_grad():
            # Direct pass (if images are huge, we might need to crop, but try direct first)
            denoised_t = model(noisy_t)

        denoised_np = tensor_to_np(denoised_t)
        clean_np_uint8 = (clean_np * 255).astype(np.uint8)

        # Metrics
        h, w, _ = clean_np_uint8.shape
        denoised_np = denoised_np[:h, :w, :] # Safety crop

        p = psnr(clean_np_uint8, denoised_np)
        s = ssim(clean_np_uint8, denoised_np, channel_axis=2)
        t = calculate_tv(denoised_np)
        
        psnr_vals.append(p)
        ssim_vals.append(s)
        tv_vals.append(t)

        # Save first 5
        if i < 5 and SAVE_DEBUG_IMAGES:
            noisy_np = tensor_to_np(noisy_t)
            combined = np.hstack((noisy_np, denoised_np, clean_np_uint8))
            cv2.imwrite(f"{PROJECT_PATH}/results/val_{i}.png", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    # 4. Results
    print("\n" + "="*40)
    print(f"📊 FINAL RESULTS")
    print(f"Avg PSNR: {sum(psnr_vals)/len(psnr_vals):.2f} dB")
    print(f"Avg SSIM: {sum(ssim_vals)/len(ssim_vals):.4f}")
    print(f"Avg TV:   {sum(tv_vals)/len(tv_vals):.4f}")
    print("="*40)

if __name__ == "__main__":
    main()