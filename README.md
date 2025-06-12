# Detail:
  - rockchip_yolonas_pth_to_onnx.py
    
    Modify From [DeepStream Convert Tool](https://github.com/marcoslucianops/DeepStream-Yolo) + Cut Post process in YOLO-NAS ONNX Model
  - export_yolonas_onnx_to_rknn.py
    
    Modify From https://github.com/MarcA711/rknn-models
  
# How to use:
** You can down input size this step.
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

  - t: SOC Type (rk3588, rk3566 etc.)
  - w: Weight Model (onnx format)
  - n: Number Classes (COCO = 80 Classes)
  - s: Input Size (320x320)
  - q: Quantize (True of False) If True, It's used dataset from datasets folder.

** If You used custom model on frigate. You need modify frigate file.
** Dont' forget install RKNN toolkit2 https://github.com/airockchip/rknn-toolkit2

**Compare Graph before and after modify**
![Screenshot from 2025-05-01 21-48-34](https://github.com/user-attachments/assets/39f49b10-db78-4857-b596-b11a382daf4d)

**Result from yolo_nas_s quantize input 640x640**
![Result_yolo_nas_s_i8_s640x640](https://github.com/user-attachments/assets/d23c11e6-9028-4cd9-8d9d-104550be0885)

**Result from yolo_nas_s quantize input 320x320**
![Result_yolo_nas_s_i8_s320x320](https://github.com/user-attachments/assets/0abe2236-5b19-46b3-a493-0a3da7373b06)

**Result from yolo_nas_s quantize input 224x224**
![Result_yolo_nas_s_i8_s224x224](https://github.com/user-attachments/assets/e87e7304-d680-4789-b8fd-fe95a544fcef)



