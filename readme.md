# SSFormer

## Developers
Jie Feng<br>

## Prerequisites
python==3.9<br> 
pytorch==1.10.01<br>
pymatgen==2023.9.25<br>
pandas==2.1.3<br> 
numpy==1.22.4<br>
matplotlib==3.8.3<br>
seaborn==0.13.2<br>
scikit-learn==1.0.1<br>
tqmd==4.66.1<br>

## Usage
###  Custom dataset  
To run the entire model, you need to customize the dataset. The pre-training dataset should include the following three parts: 2), 3), and 4). The fine-tuning dataset should include the following four parts.
#### 1) id_prop.csv: A CSV file containing the ID of each crystal structure and its corresponding attribute values.
#### 2) atom_init.json: a JSON file that stores the initialization vector for each element.
#### 3) crystal_slices.csv: a CSV file containing the SLICES string corresponding to each crystal structure.
#### 4) id.cif: a CIF file that recodes the crystal structure, where ID is the unique ID for the crystal.

###  Pre-training
#### To pre-train the model using SSL from scratch, one can run python pretrain.py.where the configurations are defined in config_multiview.yaml.<br>
`python pretrain.py
 `<br>

###  Fine-tuning
#### To fine-tune the pre-trained SSFormer, one can run finetune_transformer.py where the configurations are defined in config_ft_transformer.yaml.<br>
`python finetune_transformer.py
`<br>
#### Similarly, to fine-tune the pre-trained CGCNN, one can run finetune_cgcnn.py where the configurations are defined in config_ft_cgcnn.yaml
`python finetune_cgcnn.py
`<br>


