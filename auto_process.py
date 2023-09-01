import argparse

import torch
import torch.fx as fx
from loguru import logger

from netspresso.compressor import ModelCompressor, Task, Framework


def parse_args():
    parser = argparse.ArgumentParser()

    """
        Compression arguments
    """
    parser.add_argument('-n', '--name', type=str)

    """
        Compression arguments
    """
    parser.add_argument('-w', '--weight_path', type=str)
    parser.add_argument("--compression_method", type=str, choices=["PR_L2", "PR_GM", "PR_NN", "PR_ID", "FD_TK", "FD_CP", "FD_SVD"], default="PR_L2")
    parser.add_argument("--recommendation_method", type=str, choices=["slamp", "vbmf"], default="slamp")
    parser.add_argument("--compression_ratio", type=int, default=0.5)
    parser.add_argument("-m", "--np_email", help="NetsPresso login e-mail", type=str)
    parser.add_argument("-p", "--np_password", help="NetsPresso login password", type=str)

    """
        Fine-tuning arguments
    """
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    """ 
        Convert YOLOX model to fx 
    """
    logger.info("yolov5 to fx graph start.")

    model = torch.load(args.weight_path, map_location='cpu')

    #save model_torchfx.pt
    if isinstance(model, torch.nn.Sequential):
        model.train()
        model[-1].export = False
        model_head = model[-1]
        model = model[0]        
    else:
        model = model['model']
        model.train()
        model.model[-1].export = False
        model_head = model.model[-1]
        
        _graph = fx.Tracer().trace(model, {'augment': False, 'profile':False, 'visualize':False})
        model = fx.GraphModule(model, _graph)

    model.float()
    model_head.float()
    torch.save(model, f'{args.name}_fx.pt')
    torch.save(model_head, f'{args.name}_head_fx.pt')
    
    logger.info("yolov5 to fx graph end.")

    """
        Model compression - recommendation compression 
    """
    logger.info("Compression step start.")
    
    compressor = ModelCompressor(email=args.np_email, password=args.np_password)

    UPLOAD_MODEL_NAME = args.name
    TASK = Task.OBJECT_DETECTION
    FRAMEWORK = Framework.PYTORCH
    UPLOAD_MODEL_PATH = args.name + '_fx.pt'
    INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": (args.imgsz, args.imgsz)}]
    model = compressor.upload_model(
        model_name=UPLOAD_MODEL_NAME,
        task=TASK,
        framework=FRAMEWORK,
        file_path=UPLOAD_MODEL_PATH,
        input_shapes=INPUT_SHAPES,
    )

    COMPRESSION_METHOD = args.compression_method
    RECOMMENDATION_METHOD = args.recommendation_method
    RECOMMENDATION_RATIO = args.compression_ratio
    COMPRESSED_MODEL_NAME = f'{UPLOAD_MODEL_NAME}_{COMPRESSION_METHOD}_{RECOMMENDATION_RATIO}'
    OUTPUT_PATH = COMPRESSED_MODEL_NAME + '.pt'
    compressed_model = compressor.recommendation_compression(
        model_id=model.model_id,
        model_name=COMPRESSED_MODEL_NAME,
        compression_method=COMPRESSION_METHOD,
        recommendation_method=RECOMMENDATION_METHOD,
        recommendation_ratio=RECOMMENDATION_RATIO,
        output_path=OUTPUT_PATH,
    )

    logger.info("Compression step end.")
