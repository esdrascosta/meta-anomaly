import pdb
import torch
from torchvision import datasets, transforms as T
from datasets.mvtech import MVTecAD

if __name__ ==  '__main__':
    objs = [ 'bottle', 'cable', 'capsule', 
          'carpet', 'grid', 'hazelnut', 
          'leather', 'metal_nut', 'pill', 
          'screw', 'tile', 'toothbrush', 
          'transistor', 'wood', 'zipper'] 

    for obj in objs:

        transform = T.Compose([T.Resize(256), T.ToTensor()])

        train_set = MVTecAD(
            obj=obj,
            transform=transform
        )

        channel_means = []
        channel_std = []
        for i in range(3):
            means = []
            stds = []
            for img, _ in train_set:
                means.append(torch.mean(img[i, :, :]))
                stds.append(torch.std(img[i, :, :]))

            mean = torch.mean(torch.tensor(means))
            std = torch.mean(torch.tensor(stds))    
            channel_means.append(mean.numpy())
            channel_std.append(std.numpy())

        print(f'{obj} mean={channel_means}, std={channel_std}')