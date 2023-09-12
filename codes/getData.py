import gdown
import torch
import torchvision
import os
import pandas as pd
import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
#

class DDIData(ImageFolder):
    def __init__(self,root,transform=None):
        super(DDIData, self).__init__(os.path.join(root,"ddidiversedermatologyimages"), transform=transform)
        
        csv_path = os.path.join(root,"ddidiversedermatologyimages", "ddi_metadata.csv")          
        pd.read_csv(csv_path)
        m_key = 'malignant'
        if m_key not in self.annotations:
            self.annotations[m_key] = self.annotations['malignancy(malig=1)'].apply(lambda x: x==1)

    def __getitem__(self, index):
        img, target = super(DDIData, self).__getitem__(index)
        path = self.imgs[index][0]        
        annotation = dict(self.annotations[self.annotations.DDI_file==path.split("/")[-1]])
        target = int(annotation['malignant'].item()) # 1 if malignant, 0 if benign
        skin_tone = annotation['skin_tone'].item() # Fitzpatrick- 12, 34, or 56
        return path, img, target, skin_tone

    def subset(self, skin_tone=None, diagnosis=None):
        skin_tone = [12, 34, 56] if skin_tone is None else skin_tone
        diagnosis = ["benign", "malignant"] if diagnosis is None else diagnosis
        for si in skin_tone: 
            assert si in [12,34,56], f"{si} is not a valid skin tone"
        for di in diagnosis: 
            assert di in ["benign", "malignant"], f"{di} is not a valid diagnosis"
        indices = np.where(self.annotations['skin_tone'].isin(skin_tone) & \
                           self.annotations['malignant'].isin([di=="malignant" for di in diagnosis]))[0]
        return Subset(self, indices)