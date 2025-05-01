import os
import sys
import argparse
from rknn.api import RKNN

def parse_args():
    parser = argparse.ArgumentParser(description='ONNX to RKNN YOLO-NAS conversion')
    parser.add_argument('-t', '--soc', required=True, help='Convert Rockchip Models rk3566, rk3588 etc (required)')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.onnx) file path (required)')
    parser.add_argument('-n', '--classes', type=int, default=80, help='Number of trained classes (default 80)')
    parser.add_argument('-s', '--size', type=int, default=320, help='Inference size [H,W] (default [320])')
    parser.add_argument("-q", "--quantize", required=False,  default=False, type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')

    return args


def main(args):


    INPUT_MODEL = args.weights
    WIDTH = args.size
    HEIGHT = args.size
    OUTPUT_MODEL_BASENAME = (args.weights).split(".")[0]
    QUANTIZATION = args.quantize
    DATASET = './datasets/coco20/dataset_coco20.txt'

    # Config
    MEAN_VALUES = [[0, 0, 0]]
    STD_VALUES = [[255, 255, 255]]
    QUANT_IMG_RGB2BGR = True
    QUANTIZED_DTYPE = "asymmetric_quantized-8"
    QUANTIZED_ALGORITHM = "normal"
    QUANTIZED_METHOD = "channel"
    FLOAT_DTYPE = "float16"
    OPTIMIZATION_LEVEL = 2
    TARGET_PLATFORM = args.soc
    CUSTOM_STRING = None
    REMOVE_WEIGHT = None
    COMPRESS_WEIGHT = False
    SINGLE_CORE_MODE = False
    MODEL_PRUNNING = False
    OP_TARGET = None
    DYNAMIC_INPUT = None

    if QUANTIZATION:
        quant_suff = "-i8"
    else:
        quant_suff = ""

    OUTPUT_MODEL_FILE = "{}/{}-{}x{}{}-{}.rknn".format(args.soc, OUTPUT_MODEL_BASENAME, WIDTH, HEIGHT, quant_suff, args.soc)
    os.makedirs("{}".format(args.soc), exist_ok=True)


    rknn = RKNN()
    rknn.config(mean_values=MEAN_VALUES,
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
                dynamic_input=DYNAMIC_INPUT)

    if rknn.load_onnx(INPUT_MODEL) != 0:
        print('Error loading model.')
        exit()

    if rknn.build(do_quantization=QUANTIZATION, dataset=DATASET) != 0:
        print('Error building model.')
        exit()

    if rknn.export_rknn(OUTPUT_MODEL_FILE) != 0:
        print('Error exporting rknn model.')
        exit()




if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))

