import argparse

import torch
import torch.fx as fx
from loguru import logger


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

    torch.save(model, f'{args.name}_fx.pt')
    torch.save(model_head, f'{args.name}_head_fx.pt')
    
    logger.info("yolov5 to fx graph end.")
