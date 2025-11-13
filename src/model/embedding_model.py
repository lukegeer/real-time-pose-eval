import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseEmbeddingNet(nn.Module):
    def __init__(self, input_dims=51, hidden_dims=[256, 128], embed_dims=64, dropout=0.2, sigma=0.05):
        super().__init__()

        layers = []

        layers.append(nn.Linear(input_dims, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dims[-1], embed_dims))

        self.network = nn.Sequential(*layers)

        self.sigma = sigma

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.network(x)

        x = F.normalize(x, p=2, dim=1)

        return x
        

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, oks_weight=0.3, sigma=0.05):

        super().__init__()

        self.margin = margin
        self.oks_weight = oks_weight
        self.sigma=sigma

    def forward(self, embed1, embed2, keypoints1, keypoints2, labels):

        similarity = torch.sum(embed1 * embed2, dim=1)

        oks = self.compute_oks(keypoints1, keypoints2)


        combined_sim = (1 - self.oks_weight) * similarity + self.oks_weight * oks
        
        loss = labels * (1 - combined_sim) + (1 - labels) * torch.clamp(combined_sim - self.margin, min=0)

        loss = torch.mean(loss)

        return loss
    
    def compute_oks(self, keypoints1, keypoints2):
        xy1 = keypoints1[:, :, :2]
        xy2 = keypoints2[:, :, :2]
        conf1 = keypoints1[:, :, 2]
        conf2 = keypoints2[:, :, 2]

        xy1_normalized = self.normalize_keypoints(xy1)
        xy2_normalized = self.normalize_keypoints(xy2)


        squared_dist = torch.sum((xy1_normalized - xy2_normalized) ** 2, dim=2)

        oks = torch.exp(-squared_dist / (2 * self.sigma ** 2))

        weights = torch.minimum(conf1, conf2)


        oks_weighted = torch.sum(weights * oks, dim=1) / (torch.sum(weights, dim=1) + 1e-8)

        return oks_weighted


    def normalize_keypoints(self, xy):
        xmin = torch.min(xy[:, :, 0], dim=1)[0]
        xmax = torch.max(xy[:, :, 0], dim=1)[0]
        ymin = torch.min(xy[:, :, 1], dim=1)[0]
        ymax = torch.max(xy[:, :, 1], dim=1)[0]
        
        centerx = (xmin + xmax) / 2
        centery = (ymin + ymax) / 2
        center = torch.stack([centerx, centery], dim=1)

        width = xmax - xmin
        height = ymax - ymin
        area = width * height
        scale = torch.sqrt(area + 1e-8)


        xy_norm = (xy - center.unsqueeze(1)) / scale.unsqueeze(1).unsqueeze(2)

        return xy_norm