from __future__ import print_function, division

import csv
import functools
import  json
#import  you
import  random
import warnings
import math
import  numpy  as  np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler



class CORE_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, tokenizer, use_ratio = 1, which_label = 'void_fraction'):
            label_dict = {
                'void_fraction':2,
                'pld':3,
                'lcd':4
            }
            self.data = data[:int(len(data)*use_ratio)]
            self.mofid = self.data[:, 1].astype(str)
            self.tokens = np.array([tokenizer.encode(i, max_length=512, truncation=True,padding='max_length') for i in self.mofid])
            self.label = self.data[:, label_dict[which_label]].astype(float)
            # self.label = self.label/np.max(self.label)
            self.tokenizer = tokenizer

    def __len__(self):
            return len(self.label)
            
    @functools.lru_cache(maxsize=None) 
    def __getitem__(self, index):
            # Load data and get label
            X = torch.from_numpy(np.asarray(self.tokens[index]))
            y = torch.from_numpy(np.asarray(self.label[index])).view(-1,1)

            return X, y.float()

def get_slices_token(slice_str):
    # 定义元素周期表字典
    elements = {
        "CLS":1, 'H': 162, 'He': 163,
        'Li': 164, 'Be': 165, 'B': 166, 'C': 167, 'N': 168, 'O': 169, 'F': 170, 'Ne': 171,
        'Na': 172, 'Mg': 173, 'Al': 174, 'Si': 175, 'P': 176, 'S': 177, 'Cl': 178, 'Ar': 179,
        'K': 180, 'Ca': 181, 'Sc': 182, 'Ti': 183, 'V': 184, 'Cr': 185, 'Mn': 186, 'Fe': 187,
        'Co': 188, 'Ni': 189, 'Cu': 190, 'Zn': 191, 'Ga': 192, 'Ge': 193, 'As': 194, 'Se': 195,
        'Br': 196, 'Kr': 197, 'Rb': 198, 'Sr': 199, 'Y': 200, 'Zr': 201, 'Nb': 202, 'Mo': 203,
        'Tc': 204, 'Ru': 205, 'Rh': 206, 'Pd': 207, 'Ag': 208, 'Cd': 209, 'In': 210, 'Sn': 211,
        'Sb': 212, 'Te': 213, 'I': 214, 'Xe': 215, 'Cs': 216, 'Ba': 217, 'La': 218, 'Ce': 219,
        'Pr': 220, 'Nd': 221, 'Pm': 222, 'Sm': 223, 'Eu': 224, 'Gd': 225, 'Tb': 226, 'Dy': 227,
        'Ho': 228, 'Er': 229, 'Tm': 230, 'Yb': 231, 'Lu': 232, 'Hf': 233, 'Ta': 234, 'W': 235,
        'Re': 236, 'Os': 237, 'Ir': 238, 'Pt': 239, 'Au': 240, 'Hg': 241, 'Tl': 242, 'Pb': 243,
        'Bi': 244, 'Po': 245, 'At': 246, 'Rn': 247, 'Fr': 248, 'Ra': 249, 'Ac': 250,
        'Th': 252, 'Pa': 253, 'U': 254, 'Np': 255, 'Pu': 256, 'Am': 257, 'Cm': 258,
        'Bk': 259, 'Cf': 260, 'Es': 261, 'Fm': 262, 'Md': 263, 'No': 264, 'Lr': 265,
        "ooo": 266, "oo-": 267, "oo+": 268, "o-o": 269, "o+o": 270, "-oo": 271, "+oo": 272,
        "++o": 273, "+o+": 274, "o++": 275, "--o": 276, "-o-": 277, "o--": 278, "+o-": 279, "+-o": 280, "o+-": 281,
        "o-+": 282, "-+o": 283, "-o+": 284, "+++": 285, "---": 286, "++-": 287, "+-+": 288, "-++": 289,
        "--+": 290, "-+-": 291, "+--": 292, "SEP":293
    }

    # 分割字符串为列表
    tokens = slice_str.split()

    # 初始化结果列表
    slice_list = []

    for token in tokens:
        if token not in elements:
            slice_list.append(int(token) + 1)
        else:
            slice_list.append(elements[token])

    # 计算需要填充0的数量
    padding_length = 512 - len(tokens)
    # 检查是否需要填充
    if padding_length > 0:
        # 使用 extend() 方法来添加指定数量的0
        slice_list.extend([0] * padding_length)
    # print(slice_list)
    # print(len(slice_list))
    # print(tokens)
    # print(len(tokens))
    return slice_list

class SLICES_ID_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data):
            self.data = data
#            print(self.data)
        #     self.data = data[:int(len(data)*use_ratio)]
            self.slice_str = self.data[:, 0].astype(str)
            self.tokens = np.array([get_slices_token(s) for s in self.slice_str])
            self.label = self.data[:, 1].astype(float)

    def __len__(self):
            return len(self.label)
            
    @functools.lru_cache(maxsize=None) 
    def __getitem__(self, index):
            # Load data and get label
            X = torch.from_numpy(np.asarray(self.tokens[index]))
            y = torch.from_numpy(np.asarray(self.label[index])).view(-1,1)

            return X, y.float()


class MOF_pretrain_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, tokenizer, use_ratio = 1):

            self.data = data[:int(len(data)*use_ratio)]
            self.mofid = self.data.astype(str)
            self.tokens = np.array([tokenizer.encode(i, max_length=512, truncation=True,padding='max_length') for i in self.mofid])
            self.tokenizer = tokenizer

    def __len__(self):
            return len(self.mofid)
            
    @functools.lru_cache(maxsize=None) 
    def __getitem__(self, index):
            # Load data and get label
            X = torch.from_numpy(np.asarray(self.tokens[index]))

            return X.type(torch.LongTensor)


class MOF_tsne_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, tokenizer):
            self.data = data
            self.mofid = self.data[:, 0].astype(str)
            self.tokens = np.array([tokenizer.encode(i, max_length=512, truncation=True,padding='max_length') for i in self.mofid])
            self.label = self.data[:, 1].astype(float)

            self.tokenizer = tokenizer

    def __len__(self):
            return len(self.label)
            
    @functools.lru_cache(maxsize=None) 
    def __getitem__(self, index):
            # Load data and get label
            X = torch.from_numpy(np.asarray(self.tokens[index]))
            y = self.label[index]
            topo = self.mofid[index].split('&&')[-1].split('.')[0]
            return X, y, topo

