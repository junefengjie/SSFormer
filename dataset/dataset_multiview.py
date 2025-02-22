from __future__ import print_function, division
import pandas as pd
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
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

from dataset.augmentation import RotationTransformation, PerturbStructureTransformation, RemoveSitesTransformation


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

class MOF_ID_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, tokenizer, use_ratio = 1):

            self.data = data[:int(len(data)*use_ratio)]
            self.mofid = self.data[:, 1].astype(str)
            self.tokens = np.array([tokenizer.encode(i, max_length=512, truncation=True,padding='max_length') for i in self.mofid])
            self.label = self.data[:, 2].astype(float)

            self.tokenizer = tokenizer

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
    def __init__(self, data, tokenizer, use_ratio = 1):

            self.data = data[:int(len(data)*use_ratio)]
            self.mofname = self.data[:, 0].astype(str)
            self.mofid = self.data[:, 1].astype(str)
            self.tokens = np.array([tokenizer.encode(i, max_length=512, truncation=True,padding='max_length') for i in self.mofid])
            self.label = self.data[:, 2].astype(float)

            self.tokenizer = tokenizer

    def __len__(self):
            return len(self.label)
            
    @functools.lru_cache(maxsize=None) 
    def __getitem__(self, index):
            # Load data and get label
            X = torch.from_numpy(np.asarray(self.tokens[index]))

            return X, self.label[index], self.mofname[index], self.mofid[index]


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
            slice_list.append(int(token))
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





def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, val_ratio=0.1, random_seed = 11, num_workers=1, 
                              pin_memory=False, **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.
    !!! The dataset needs to be shuffled before using the function !!!
    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool
    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    train_ratio = 1 - val_ratio
    train_size = int(train_ratio * total_size)
    valid_size = int(val_ratio * total_size)
    indices = list(range(total_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(indices[train_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers, drop_last=True,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers, drop_last=True,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    return train_loader, val_loader


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.
    Parameters
    ----------
    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)
      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int
    Returns
    -------
    N = sum(n_i); N0 = sum(i)
    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_tokens = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), tokens, cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_tokens.append(tokens)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx),\
        torch.cat(batch_tokens, dim=0),\
        batch_cif_ids


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------
        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array
        Parameters
        ----------
        distance: np.array shape n-d array
          A distance matrix of any shape
        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.
    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class  AtomCustomJSONInitializer ( AtomInitializer ):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.
    Parameters
    ----------
    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:
    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...
    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.
    atom_init.json: a JSON file that stores the initialization vector for each
    element.
    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.
    Parameters
    ----------
    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset
    Returns
    -------
    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123):
        self.root_dir  =  root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'

        id_prop_file = os.path.join(self.root_dir, 'mp_to_slice_filtered.csv')
        assert os.path.exists(id_prop_file), 'mp_to_slice_filtered.csv does not exist!'        
        self.id_prop_data = self.id_prop_data = pd.read_csv(id_prop_file)
        print(self.id_prop_data)
        print(type(self.id_prop_data))
        
        atom_init_file = os.path.join(self.root_dir,'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    #@functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        #print(self.id_prop_data[idx])
        row = self.id_prop_data.iloc[idx]
        cif_id = row['filename']  # 获取 ID
        slice_token = row['slices_string']  # 获取 slice 字符串
        fname = cif_id
        if fname[-4:] != '.cif':
            fname = fname + '.cif'
        crys = Structure.from_file(os.path.join(self.root_dir, fname))

        tokens = np.array([get_slices_token(slice_token)])
        
        tokens = torch.from_numpy(tokens)
#        print(tokens.shape)
#        print(tokens)
        crystal = crys.copy()

        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                               for  i  in  range ( len ( crystal ))])
        atom_fea = torch.Tensor(atom_fea)

        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx , nbr_fea  = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)

        nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        # target = torch.Tensor([float(target)])

        return (atom_fea, nbr_fea, nbr_fea_idx), tokens, cif_id
 