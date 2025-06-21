import os
import sys
import argparse
from rknn.api import RKNN

def parse_args():
    parser = argparse.ArgumentParser(description='ONNX to RKNN YOLO-NAS conversion')
    parser.add_argument('-t', '--soc', required=True, help='Convert Rockchip Models rk3566, rk3588 etc (required)')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.onnx) file path (required)')
    parser.add_argument('-n', '--classes', type=int, default=80, help='Number of trained classes (default 80)')
    parser.add_argument('--width', type=int, default=640, help='Input width (default 640)')
    parser.add_argument('--height', type=int, default=320, help='Input height (default 320)')
    parser.add_argument("-q", "--quantize", required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--dataset', default='./datasets/coco20/dataset_coco20.txt', help='Dataset for quantization')
    args = parser.parse_args()
    
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    
    if args.quantize and not os.path.isfile(args.dataset):
        print(f"Warning: Dataset file {args.dataset} not found. Quantization may fail.")

    return args


def main(args):
    INPUT_MODEL = args.weights
    WIDTH = args.width
    HEIGHT = args.height
    OUTPUT_MODEL_BASENAME = os.path.splitext(os.path.basename(args.weights))[0]
    QUANTIZATION = args.quantize
    DATASET = args.dataset

    # YOLO-NAS specific config - normalized inputs (0-1 range)
    MEAN_VALUES = [[0, 0, 0]]
    STD_VALUES = [[1, 1, 1]]  # Changed from [255,255,255] to [1,1,1] for normalized inputs
    QUANT_IMG_RGB2BGR = True
    QUANTIZED_DTYPE = "asymmetric_quantized-8"
    QUANTIZED_ALGORITHM = "normal"
    QUANTIZED_METHOD = "channel"
    FLOAT_DTYPE = "float16"
    OPTIMIZATION_LEVEL = 3  # Increased from 2 to 3 for better optimization
    TARGET_PLATFORM = args.soc
    CUSTOM_STRING = None
    REMOVE_WEIGHT = None
    COMPRESS_WEIGHT = True  # Changed to True for smaller model size
    SINGLE_CORE_MODE = False
    MODEL_PRUNNING = False
    OP_TARGET = None
    DYNAMIC_INPUT = None

    # Output file naming
    if QUANTIZATION:
        quant_suff = "-i8"
    else:
        quant_suff = ""

    OUTPUT_MODEL_FILE = "{}/{}-{}x{}{}-{}.rknn".format(
        args.soc, OUTPUT_MODEL_BASENAME, WIDTH, HEIGHT, quant_suff, args.soc
    )
    os.makedirs("{}".format(args.soc), exist_ok=True)

    print(f"Converting {INPUT_MODEL} to {OUTPUT_MODEL_FILE}")
    print(f"Input size: {WIDTH}x{HEIGHT}")
    print(f"Quantization: {'Enabled' if QUANTIZATION else 'Disabled'}")
    print(f"Target platform: {TARGET_PLATFORM}")

    # Initialize RKNN
    rknn = RKNN(verbose=True)
    
    # Configuration
    ret = rknn.config(
        mean_values=MEAN_VALUES,
        std_values=STD_VALUES,
        quant_img_RGB2BGR=QUANT_IMG_RGB2BGR,
        quantized_dtype=QUANTIZED_DTYPE,
        quantized_algorithm=QUANTIZED_ALGORITHM,
        quantized_method=QUANTIZED_METHOD,
        float_dtype=FLOAT_DTYPE,
        optimization_level=OPTIMIZATION_LEVEL,
        target_platform=TARGET_PLATFORM,
        custom_string=CUSTOM_STRING,
        remove_weight=REMOVE_WEIGHT,
        compress_weight=COMPRESS_WEIGHT,
        single_core_mode=SINGLE_CORE_MODE,
        model_pruning=MODEL_PRUNNING,
        op_target=OP_TARGET,
        dynamic_input=DYNAMIC_INPUT
    )
    
    if ret != 0:
        print('Error in RKNN config.')
        return -1

    # Load ONNX model
    print('Loading ONNX model...')
    ret = rknn.load_onnx(INPUT_MODEL)
    if ret != 0:
        print('Error loading ONNX model.')
        rknn.release()
        return -1

    # Build model
    print('Building RKNN model...')
    if QUANTIZATION and os.path.isfile(DATASET):
        ret = rknn.build(do_quantization=QUANTIZATION, dataset=DATASET)
    else:
        if QUANTIZATION:
            print("Warning: Quantization requested but dataset not found. Building without quantization.")
        ret = rknn.build(do_quantization=False)
    
    if ret != 0:
        print('Error building RKNN model.')
        rknn.release()
        return -1

    # Export RKNN model
    print('Exporting RKNN model...')
    ret = rknn.export_rknn(OUTPUT_MODEL_FILE)
    if ret != 0:
        print('Error exporting RKNN model.')
        rknn.release()
        return -1

    # Clean up
    rknn.release()
    
    print(f'✅ Successfully converted to: {OUTPUT_MODEL_FILE}')
    print(f'Model size: {os.path.getsize(OUTPUT_MODEL_FILE) / (1024*1024):.2f} MB')
    
    return 0


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))