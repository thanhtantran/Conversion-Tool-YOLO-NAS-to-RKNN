# Detail:
  - rockchip_yolonas_pth_to_onnx.py
    
    Modify From DeepStream Convert Tool + Cut Post process in YOLO-NAS ONNX Model
  - export_yolonas_onnx_to_rknn.py
    
    Modify From https://github.com/MarcA711/rknn-models
  
# How to use:
- Convert pth model to onnx (yolo_nas_s, input size 320x320, coco_label)
  
  Run:  python rockchip_yolonas_pth_to_onnx.py -m yolo_nas_s -w last.pth  --simplify -n 80 -s 320
- Convert pth model to onnx (yolo_nas_s, input size 640x640 downsize model to support 320x320, coco_label)
  
  Run:  python rockchip_yolonas_pth_to_onnx.py -m yolo_nas_s -w last.pth  --simplify -n 80 -s 320

  - m: Base Model (yolo_nas_s, yolo_nas_m, yolo_nas_l)
  - w: Weight Model
  - n: Number Classes
  - s: Input Size

- Convert onnx to rknn
  
  Run: python export_yolonas_onnx_to_rknn.py -t rk3588 -w rk_last.onnx -n 80 -s 320
  
  Run: python export_yolonas_onnx_to_rknn.py -t rk3588 -w rk_last.onnx -n 80 -s 320 -q true

  - t SOC Type (rk3588, rk3566 etc.)
  - w Weight Model (onnx format)
  - n Number Classes (COCO = 80 Classes)
  - s Input Size (320x320)
  - q Quantize (True of False) If True, It's used dataset from datasets folder.

** If You used custom model on frigate. You need modify frigate file.
