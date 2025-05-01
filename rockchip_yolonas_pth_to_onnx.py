import os
import sys
import argparse
import warnings
import onnx
import torch
import torch.nn as nn
from super_gradients.training import models
import onnx_graphsurgeon as gs
import numpy as np

#Example Run
#python rockchip_yolonas_pth_to_onnx.py -m yolo_nas_s -w latest.pth --simplify -n 80 -s 320
#python rockchip_yolonas_pth_to_onnx.py -m yolo_nas_s -w latest.pth --simplify -n 80 -s 640

class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        boxes = x[0]
        scores, classes = torch.max(x[1], 2, keepdim=True)
        classes = classes.float()
        return boxes, scores, classes

def mod_yolonas_onnx_model(
    input_path: str = "deepstream_output.onnx",
    output_path: str = "deepstream_output_rk.onnx",
    sigmoid_node_name: str = "/0/heads/Sigmoid",
    bbox_tensor_name: str = "boxes",
    new_scores_name: str = "scores"
):

    print(f"🔍 Loading model: {input_path}")
    model = onnx.load(input_path)
    graph = gs.import_onnx(model)

    # Remove post-processing ops
    post_ops = {"ArgMax", "Cast", "ReduceMax", "TopK"}
    graph.nodes = [node for node in graph.nodes if node.op not in post_ops]

    # Find sigmoid node
    sigmoid_node = next((node for node in graph.nodes if node.op == "Sigmoid" and sigmoid_node_name in node.name), None)
    if not sigmoid_node:
        raise RuntimeError(f"Sigmoid node '{sigmoid_node_name}' not found.")
    scores_tensor = sigmoid_node.outputs[0]
    scores_tensor.name = new_scores_name

    # Get bbox tensor
    bbox_tensor = graph.tensors().get(bbox_tensor_name)
    if not bbox_tensor:
        raise RuntimeError(f"BBox tensor '{bbox_tensor_name}' not found.")

    # Clear any previous output links
    for tname in [new_scores_name, "classes"]:
        tensor = graph.tensors().get(tname)
        if tensor:
            tensor.outputs.clear()

    # Redefine outputs
    graph.outputs.clear()
    bbox_tensor.outputs.clear()
    scores_tensor.outputs.clear()
    graph.outputs.extend([bbox_tensor, scores_tensor])

    # Finalize
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), output_path)
    print(f"✅ Clean model saved: {output_path}")
    
def suppress_warnings():
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)


def yolonas_export(model_name, weights, num_classes, size):
    img_size = size * 2 if len(size) == 1 else size
    model = models.get(model_name, num_classes=num_classes, checkpoint_path=weights)
    model.eval()
    model.prep_model_for_conversion(input_size=[1, 3, *img_size])
    return model


def main(args):
    suppress_warnings()

    print('\nStarting: %s' % args.weights)

    print('Opening YOLO-NAS model\n')

    device = torch.device('cpu')
    model = yolonas_export(args.model, args.weights, args.classes, args.size)

    model = nn.Sequential(model, DeepStreamOutput())

    img_size = args.size * 2 if len(args.size) == 1 else args.size

    onnx_input_im = torch.zeros(args.batch, 3, *img_size).to(device)
    onnx_output_file = os.path.basename(args.weights).split('.pt')[0] + '.onnx'

    dynamic_axes = {
        'input': {
            0: 'batch'
        },
        'boxes': {
            0: 'batch'
        },
        'scores': {
            0: 'batch'
        },
        'classes': {
            0: 'batch'
        }
    }

    print('\nExporting the model to ONNX')
    torch.onnx.export(model, onnx_input_im, onnx_output_file, verbose=False, opset_version=args.opset,
                      do_constant_folding=True, input_names=['input'], output_names=['boxes', 'scores', 'classes'],
                      dynamic_axes=dynamic_axes if args.dynamic else None)

    if args.simplify:
        print('Simplifying the ONNX model')
        import onnxsim
        model_onnx = onnx.load(onnx_output_file)
        model_onnx, _ = onnxsim.simplify(model_onnx)
        onnx.save(model_onnx, onnx_output_file)

    print('Done: %s\n' % onnx_output_file)
    
    mod_yolonas_onnx_model(
        input_path=f"{onnx_output_file}",
        output_path=f"rk_{onnx_output_file}",
        sigmoid_node_name="/0/heads/Sigmoid",
        bbox_tensor_name="boxes",
        new_scores_name="scores"
    )    

def parse_args():
    parser = argparse.ArgumentParser(description='DeepStream YOLO-NAS conversion')
    parser.add_argument('-m', '--model', required=True, help='Model name (required)')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pth) file path (required)')
    parser.add_argument('-n', '--classes', type=int, default=80, help='Number of trained classes (default 80)')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=[640], help='Inference size [H,W] (default [640])')
    parser.add_argument('--opset', type=int, default=14, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='ONNX simplify model')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic batch-size')
    parser.add_argument('--batch', type=int, default=1, help='Implicit batch-size')
    args = parser.parse_args()
    if args.model == '':
        raise SystemExit('Invalid model name')
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    if args.dynamic and args.batch > 1:
        raise SystemExit('Cannot set dynamic batch-size and implicit batch-size at same time')
    return args


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
