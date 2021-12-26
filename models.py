import torch
from torch import nn
import torchvision
from torchvision import models
import numpy as np
import sys
import torch.nn.functional as F
import scipy.spatial as sp
from utils import init_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


def cosdist(vectors):
    vectors_t = torch.t(vectors)
    distance_matrix = 1 - sp.distance.cdist(vectors, vectors_t, 'cosine')
    return distance_matrix


class API_Net(nn.Module):
    def __init__(self, num_classes=5, model_name='res101', weight_init_zero=False):
        super(API_Net, self).__init__()

        # ---------Resnet101---------
        if model_name == 'res101':
            model = models.resnet101(pretrained=True)
        # layers = list(resnet101.children())[:-2]

        # ---------Efficientnet---------
        elif model_name == 'effb0':
            model = models.efficientnet_b0(pretrained=True)
        elif model_name == 'effb1':
            model = models.efficientnet_b1(pretrained=True)
        elif model_name == 'effb2':
            model = models.efficientnet_b2(pretrained=True)
        elif model_name == 'effb3':
            model = models.efficientnet_b3(pretrained=True)
        elif model_name == 'effb4':
            model = models.efficientnet_b4(pretrained=True)
        elif model_name == 'effb5':
            model = models.efficientnet_b5(pretrained=True)
        elif model_name == 'effb6':
            model = models.efficientnet_b6(pretrained=True)
        elif model_name == 'effb7':
            model = models.efficientnet_b7(pretrained=True)
        else:
            sys.exit('wrong model name baby')

        if weight_init_zero:
            model.apply(init_weights)
            print('init weight 0')

        layers = list(model.children())[:-2]
        if 'res' in model_name:
            fc_size = model.fc.in_features
        elif 'eff' in model_name:
            fc_size = model.classifier[1].in_features
        else:
            sys.exit('wrong network name baby')

        self.conv = nn.Sequential(*layers)
        self.avg = nn.AvgPool2d(kernel_size=14, stride=1)

        self.map1 = nn.Linear(fc_size * 2, 512)
        self.map2 = nn.Linear(512, fc_size)
        self.fc = nn.Linear(fc_size, num_classes)

        self.drop = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()


    def forward(self, images, targets=None, flag='train', dist_type='euclidean'):
        # print(f'images {images.shape}')
        conv_out = self.conv(images)
        # print(f'conv_out {conv_out.shape}')
        pool_out = self.avg(conv_out)
        # print(f'pool_out {pool_out.shape}')
        pool_out = pool_out.squeeze()
        # print(f'pool_out {pool_out.shape}')

        if flag == 'train':
            intra_pairs, inter_pairs, intra_labels, inter_labels = self.get_pairs(pool_out, targets, dist_type)

            features1 = torch.cat([pool_out[intra_pairs[:, 0]], pool_out[inter_pairs[:, 0]]], dim=0)
            features2 = torch.cat([pool_out[intra_pairs[:, 1]], pool_out[inter_pairs[:, 1]]], dim=0)
            labels1 = torch.cat([intra_labels[:, 0], inter_labels[:, 0]], dim=0)
            labels2 = torch.cat([intra_labels[:, 1], inter_labels[:, 1]], dim=0)
            mutual_features = torch.cat([features1, features2], dim=1)
            map1_out = self.map1(mutual_features)
            map2_out = self.drop(map1_out)
            map2_out = self.map2(map2_out)

            gate1 = torch.mul(map2_out, features1)
            gate1 = self.sigmoid(gate1)

            gate2 = torch.mul(map2_out, features2)
            gate2 = self.sigmoid(gate2)

            features1_self = torch.mul(gate1, features1) + features1
            features1_other = torch.mul(gate2, features1) + features1

            features2_self = torch.mul(gate2, features2) + features2
            features2_other = torch.mul(gate1, features2) + features2

            logit1_self = self.fc(self.drop(features1_self))
            logit1_other = self.fc(self.drop(features1_other))
            logit2_self = self.fc(self.drop(features2_self))
            logit2_other = self.fc(self.drop(features2_other))

            return logit1_self, logit1_other, logit2_self, logit2_other, labels1, labels2

        elif flag == 'val':
            return self.fc(pool_out)
        elif flag == 'test':
            return self.fc(pool_out)


    def get_pairs(self, embeddings, labels, dist_type):
        if dist_type == 'euclidean':
            distance_matrix = pdist(embeddings).detach().cpu().numpy()
        elif dist_type == 'cosine':
            distance_matrix = cosdist(embeddings).detach().cpu().numpy()

        labels = labels.detach().cpu().numpy().reshape(-1,1)
        num = labels.shape[0]
        dia_inds = np.diag_indices(num)
        lb_eqs = (labels == labels.T)
        lb_eqs[dia_inds] = False
        dist_same = distance_matrix.copy()
        dist_same[lb_eqs == False] = np.inf
        intra_idxs = np.argmin(dist_same, axis=1)

        dist_diff = distance_matrix.copy()
        lb_eqs[dia_inds] = True
        dist_diff[lb_eqs == True] = np.inf
        inter_idxs = np.argmin(dist_diff, axis=1)

        intra_pairs = np.zeros([embeddings.shape[0], 2])
        inter_pairs  = np.zeros([embeddings.shape[0], 2])
        intra_labels = np.zeros([embeddings.shape[0], 2])
        inter_labels = np.zeros([embeddings.shape[0], 2])
        for i in range(embeddings.shape[0]):
            intra_labels[i, 0] = labels[i]
            intra_labels[i, 1] = labels[intra_idxs[i]]
            intra_pairs[i, 0] = i
            intra_pairs[i, 1] = intra_idxs[i]

            inter_labels[i, 0] = labels[i]
            inter_labels[i, 1] = labels[inter_idxs[i]]
            inter_pairs[i, 0] = i
            inter_pairs[i, 1] = inter_idxs[i]

        intra_labels = torch.from_numpy(intra_labels).long().to(device)
        intra_pairs = torch.from_numpy(intra_pairs).long().to(device)
        inter_labels = torch.from_numpy(inter_labels).long().to(device)
        inter_pairs = torch.from_numpy(inter_pairs).long().to(device)

        return intra_pairs, inter_pairs, intra_labels, inter_labels



















