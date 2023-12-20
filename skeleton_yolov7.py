import torch

import sys

sys.path.insert(0, 'yolov7')
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint


class skeleton():
    def __init__(self, path_weights:str,) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = attempt_load(path_weights, map_location=self.device).eval()
    
    def get_keypoint(self, img):
        with torch.no_grad():
            pred, _ = self.model(img)
        
        pred = non_max_suppression_kpt(pred, 
                               conf_thres=0.25, 
                               iou_thres=0.65, 
                               nc=self.model.yaml['nc'], 
                               nkpt=self.model.yaml['nkpt'], 
                               kpt_label=True)

        pred = output_to_keypoint(pred)
        

        return pred
    
