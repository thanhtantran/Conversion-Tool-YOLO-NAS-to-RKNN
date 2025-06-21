from super_gradients.training import models
import inspect

# Load YOLO-NAS small model with COCO pretrained weights
model = models.get("yolo_nas_s", pretrained_weights="coco")

# Check the export method signature
print("Export method signature:")
print(inspect.signature(model.export))

# Try the correct export approach based on SuperGradients documentation
try:
    # Method 1: Basic export (let's see what parameters it actually accepts)
    model.export(
        "yolo_nas_s_640x320.onnx"
    )
    print("✅ Basic export successful")
except Exception as e:
    print(f"Basic export failed: {e}")
    
    try:
        # Method 2: Using prep_model_for_conversion + export
        # First prepare the model
        model.eval()
        model.prep_model_for_conversion(input_size=(1, 3, 320, 640))
        
        # Then export
        model.export("yolo_nas_s_640x320.onnx")
        print("✅ Export with prep successful")
    except Exception as e2:
        print(f"Export with prep failed: {e2}")
        
        # Method 3: Check help for export method
        try:
            help(model.export)
        except:
            print("Could not get help for export method")

print("Script completed")