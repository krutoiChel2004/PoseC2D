import torch
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn

import numpy as np

from PIL import Image
import cv2

from skeleton_yolov7 import skeleton
from tools import get_skeleton_img
from data import YogaPoses
import albumentations as A

class PoseC2D(nn.Module):
    def __init__(self, num_channels = 3, weights_bakbone = None):
        super(PoseC2D, self).__init__()
        self.skeleton_model = skeleton(path_weights='weights\detector\yolov7-w6-pose.pt')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bakbone = models.resnet50(pretrained=True).to(self.device)
        # self.bakbone.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bakbone.fc = nn.Linear(2048, 5)

        if weights_bakbone is not None:
            self.bakbone.load_state_dict(torch.load(weights_bakbone))
    
    def forward(self, x):
        x = torch.chunk(x, x.shape[0], dim=0)
        batch = None
        for img in x:
            
            key_point = self.skeleton_model.get_keypoint(img)
            heat_map:np.array = get_skeleton_img((640, 640), key_point, 0.2, 3)
            heat_map = np.transpose(heat_map, (2, 0, 1))
            heat_map = np.expand_dims(heat_map, 0)

            heat_map_tensor = torch.from_numpy(heat_map)
            if batch is None:
                batch = heat_map_tensor
            else:
                batch = torch.cat((batch, heat_map_tensor), 0)
            # print(heat_map.shape)
            
        # print(batch.shape)

        # img_ = transforms.ToTensor()(batch)
        # img_ = torch.unsqueeze(img_, 0)
        batch = batch.to(self.device).float()
        out = self.bakbone(batch)
        return out

