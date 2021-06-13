import pdb
import torch
from torch import nn
import torchvision.models as M
import torch.nn.functional as F
from torch.nn.parameter import Parameter
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.backbone = M.resnet50(num_classes=512)
        # self.backbone.fc = nn.Identity()
        self.backbone_out_dim = 512

        self.center =  Parameter(torch.Tensor(self.backbone_out_dim))
        self.center.data.fill_(1.0)

    def init_center(self, train_set, eps=0.1):

        train_loader = torch.utils.data.DataLoader(train_set)
        n_samples = 0
        
        center = torch.ones(self.backbone_out_dim, device='cuda')

        self.eval()

        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _ = data
                inputs = inputs.to('cuda')
                outputs = self.backbone(inputs)
                n_samples += outputs.shape[0]
                center += torch.sum(outputs, dim=0)

        center = center / n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        center[(abs(center) < eps) & (center < 0)] = -eps
        center[(abs(center) < eps) & (center > 0)] = eps
        self.center = F.normalize(center, dim=-1, p=2)

    def forward(self, x):
        outputs = self.backbone(x)
        outputs = F.normalize(outputs, dim=-1, p=2)
        dist = torch.sum((outputs - self.center) ** 2, dim=1)
        # loss = torch.mean(dist)
        return dist


