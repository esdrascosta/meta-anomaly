# -*- coding: utf-8 -*-
import os
import shutil
import tarfile
from PIL import Image
import urllib.request as request
from contextlib import closing
from torch.utils.data import Dataset
# import torchvision 
import random
from PIL import Image

class MVTecAD(Dataset):

    DATASET_URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
    TAR_FILE = 'data/mvtech_ad/mvtec_anomaly_detection.tar.xz'
    DATASET_DIR = 'data/mvtech_ad'

    def __init__(self, transform=None, target_transform=None, download=False, mode='train', obj=None):
        self.data = []
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.obj = obj
        
        if download and not self._check_download():
            self._download()

        self._parse_data_dir()
        random.shuffle(self.data)

    def _parse_data_dir(self):
        temp_dataset_dir = os.path.join(self.DATASET_DIR)
        objects = [ d for d in os.listdir(temp_dataset_dir) if os.path.isdir(os.path.join(temp_dataset_dir, d)) ]
        
        # filter the disired object
        if self.obj:
            objects = [ obj for obj in objects if obj == self.obj ]

        for obj in objects:
            mode_path = os.path.join(temp_dataset_dir, obj, self.mode)
            dis = [ di for di in os.listdir(mode_path) if os.path.isdir(os.path.join(mode_path, di)) ]
            for di in dis:
                di_path = os.path.join(mode_path, di)
                for f in os.listdir(di_path):
                    if os.path.isfile(os.path.join(di_path, f)):
                        self.data.append(
                            ( os.path.join(di_path, f), 0 if di == 'good' else 1 )
                        )

    def _download(self):
        os.makedirs(self.DATASET_DIR, exist_ok=True )
        try:
            with closing(request.urlopen(self.DATASET_URL)) as r:
                with open(self.TAR_FILE, 'wb') as f:
                    shutil.copyfileobj(r, f)
        except:
            print('Error ocour when download the dataset! try download manually.')
       
        
        print(f"Extracting: {self.TAR_FILE}")

        with tarfile.open(self.TAR_FILE) as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, self.DATASET_DIR)
        os.unlink(self.TAR_FILE)

    def _check_download(self):
        return (os.path.exists(self.TAR_FILE) and os.path.isfile(self.TAR_FILE)) or \
               (os.path.exists(self.DATASET_DIR) and os.path.isdir(self.DATASET_DIR))

    def _image_loader(self, img_path):
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):

        path, target = self.data[index]
        sample = self._image_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data = MVTecAD()
    print(len(data))
    sample, target = data[0]

    plt.imshow(sample)
    plt.savefig('test.png')
