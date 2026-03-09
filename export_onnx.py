import torch
import torch.onnx
from mfdnet import MFDNet

MODEL_PATH = "best_model.pth"
ONNX_PATH = "mfdnet.onnx"

def main():
    # 1. Load Model
    device = torch.device("cpu")
    model = MFDNet(in_ch=3, base_ch=32, num_blocks=4).to(device)
    
    # Load weights
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    # 2. Dummy Input (Fixed 256x256)
    dummy_input = torch.randn(1, 3, 256, 256, device=device)

    # 3. Export (STATIC SHAPE)
    print(f"⏳ Exporting STATIC model to {ONNX_PATH}...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
        # REMOVED dynamic_axes dictionary completely!
        # This forces the model to be exactly 1x3x256x256 forever.
    )
    print(f"✅ Success! Created static '{ONNX_PATH}'")

if __name__ == "__main__":
    main()