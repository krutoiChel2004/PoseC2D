import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import sys
import cv2

sys.path.insert(0, 'yolov7')
from utils.datasets import letterbox

class YogaPoses(Dataset):
    def __init__(self, path_list:list[str], transform=None):
        super().__init__()
        self.transform = transform
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.path_Downdog = path_list[0]
        self.path_Goddess = path_list[1]
        self.path_Plank = path_list[2]
        self.path_Tree = path_list[3]
        self.path_Warrior2 = path_list[4]

        self.Downdog_list = sorted(os.listdir(self.path_Downdog))
        self.Goddess_list = sorted(os.listdir(self.path_Goddess))
        self.Plank_list = sorted(os.listdir(self.path_Plank))
        self.Tree_list = sorted(os.listdir(self.path_Tree))
        self.Warrior2_list = sorted(os.listdir(self.path_Warrior2))

    def __len__(self):
        return len(self.Downdog_list) + len(self.Goddess_list) + len(self.Plank_list) + len(self.Tree_list) + len(self.Warrior2_list)
    
    def __getitem__(self, idx):
        if idx < len(self.Downdog_list):
            img_path = os.path.join(self.path_Downdog, self.Downdog_list[idx])
            class_id = 0
        elif idx < len(self.Downdog_list) + len(self.Goddess_list):
            img_path = os.path.join(self.path_Goddess, self.Goddess_list[idx - len(self.Downdog_list)])
            class_id = 1
        elif idx < len(self.Downdog_list) + len(self.Goddess_list) + len(self.Plank_list):
            img_path = os.path.join(self.path_Plank, self.Plank_list[idx - len(self.Downdog_list) - len(self.Goddess_list)])
            class_id = 2
        elif idx < len(self.Downdog_list) + len(self.Goddess_list) + len(self.Plank_list) + len(self.Tree_list):
            img_path = os.path.join(self.path_Tree, self.Tree_list[idx - len(self.Downdog_list) - len(self.Goddess_list) - len(self.Plank_list)])
            class_id = 3
        else:
            img_path = os.path.join(self.path_Warrior2, self.Warrior2_list[idx - len(self.Downdog_list) - len(self.Goddess_list) - len(self.Plank_list) - len(self.Tree_list)])
            class_id = 4

        
        
        img = np.array(Image.open(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed["image"]
        img = letterbox(img, 640, stride=64, auto=True)[0]
        img_ = transforms.ToTensor()(img)
        #img_ = torch.unsqueeze(img_, 0)
        img_ = img_.to(self.device).float()

        return {'img':img_, 'class':class_id}
