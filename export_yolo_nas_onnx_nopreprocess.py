#!/usr/bin/env python3

import torch
import torch.onnx
import sys
import os

# Add minimal imports to avoid dependency conflicts
try:
    from super_gradients.training import models
    print("✅ SuperGradients imported successfully")
except Exception as e:
    print(f"❌ Failed to import SuperGradients: {e}")
    sys.exit(1)

def export_yolo_nas():
    try:
        print("Loading YOLO-NAS model...")
        # Load model
        model = models.get("yolo_nas_s", pretrained_weights="coco")
        model.eval()
        print("✅ Model loaded successfully")
        
        # Create dummy input tensor
        dummy_input = torch.randn(1, 3, 320, 640)
        print(f"✅ Created dummy input: {dummy_input.shape}")
        
        # Simple export approach - let SuperGradients handle it
        print("Attempting SuperGradients export...")
        try:
            model.export("yolo_nas_s_640x320_simple.onnx")
            print("✅ SuperGradients export successful: yolo_nas_s_640x320_simple.onnx")
            return True
        except Exception as e:
            print(f"SuperGradients export failed: {e}")
        
        # Fallback: Direct PyTorch ONNX export with minimal settings
        print("Attempting PyTorch ONNX export...")
        try:
            torch.onnx.export(
                model,
                dummy_input,
                "yolo_nas_s_640x320_torch.onnx",
                export_params=True,
                opset_version=11,  # Use older opset for compatibility
                do_constant_folding=False,  # Disable to avoid optimization issues
                input_names=['input'],
                output_names=['output']
            )
            print("✅ PyTorch ONNX export successful: yolo_nas_s_640x320_torch.onnx")
            return True
        except Exception as e:
            print(f"PyTorch ONNX export failed: {e}")
        
        # Last resort: Try with even simpler settings
        print("Attempting basic PyTorch export...")
        try:
            torch.onnx.export(
                model,
                dummy_input,
                "yolo_nas_s_640x320_basic.onnx",
                opset_version=9  # Very basic opset
            )
            print("✅ Basic PyTorch export successful: yolo_nas_s_640x320_basic.onnx")
            return True
        except Exception as e:
            print(f"Basic PyTorch export also failed: {e}")
            
        return False
        
    except Exception as e:
        print(f"❌ Export failed with error: {e}")
        return False

if __name__ == "__main__":
    print("=== YOLO-NAS ONNX Export Tool ===")
    success = export_yolo_nas()
    
    if success:
        print("\n✅ Export completed successfully!")
        print("Files created:")
        for filename in ["yolo_nas_s_640x320_simple.onnx", "yolo_nas_s_640x320_torch.onnx", "yolo_nas_s_640x320_basic.onnx"]:
            if os.path.exists(filename):
                size_mb = os.path.getsize(filename) / (1024 * 1024)
                print(f"  - {filename} ({size_mb:.2f} MB)")
    else:
        print("\n❌ All export methods failed.")
        print("\nTroubleshooting suggestions:")
        print("1. Check PyTorch and TorchVision compatibility:")
        print("   pip list | grep torch")
        print("2. Try reinstalling with compatible versions:")
        print("   pip install torch==1.13.1 torchvision==0.14.1")
        print("3. Or try the conda environment approach")
        sys.exit(1)