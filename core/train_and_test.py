import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
torch.manual_seed(8)

import time
import numpy as np
import argparse
import pickle
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.nn.Module.dump_patches = True
import copy
import pandas as pd
from model import save_smiles_dicts, get_smiles_dicts, get_smiles_array, Descriptor

from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import rdMolDescriptors, MolSurf
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import seaborn as sns; sns.set()

class Experiment:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.p_dropout = args.dropout
        self.epochs = args.epochs
        self.hidden_desc_len = args.hidden_desc_len
        self.learning_rate = args.lr
        self.radius = args.radius
        self.T = args.t
        self.weight_decay = args.weight_decay
        self.fingerprint_dim = args.desc_dim


    def train(model, dataset, optimizer, loss_function):
        model.train()
        np.random.seed(epoch)
        valList = np.arange(0,dataset.shape[0])
        np.random.shuffle(valList)
        batch_list = []
        for i in range(0, dataset.shape[0], batch_size):
            batch = valList[i:i+batch_size]
            batch_list.append(batch)   
        for counter, batch in enumerate(batch_list):
            batch_df = dataset.loc[batch,:]
            smiles_list = batch_df.cano_smiles.values
            y_val = batch_df[tasks[0]].values
        
            x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list, x_descriptors = get_smiles_array(smiles_list,feature_dicts)
            atoms_prediction, mol_prediction = model(torch.Tensor(x_atom),torch.Tensor(x_bonds),torch.cuda.LongTensor(x_atom_index),torch.cuda.LongTensor(x_bond_index),
                                                                            torch.Tensor(x_mask), torch.cuda.FloatTensor(x_descriptors))
        
            optimizer.zero_grad()
            loss = loss_function(mol_prediction, torch.Tensor(y_val).view(-1,1))     
            loss.backward()
            optimizer.step()

    def eval(model, dataset):
        model.eval()
        test_MAE_list = []
        test_MSE_list = []
        valList = np.arange(0,dataset.shape[0])
        batch_list = []
        for i in range(0, dataset.shape[0], batch_size):
            batch = valList[i:i+batch_size]
            batch_list.append(batch) 
        for counter, batch in enumerate(batch_list):
            batch_df = dataset.loc[batch,:]
            smiles_list = batch_df.cano_smiles.values
            y_val = batch_df[tasks[0]].values
        
            x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list, x_descriptors = get_smiles_array(smiles_list,feature_dicts)
            atoms_prediction, mol_prediction = model(torch.Tensor(x_atom),torch.Tensor(x_bonds),torch.cuda.LongTensor(x_atom_index),torch.cuda.LongTensor(x_bond_index),
                                                                            torch.Tensor(x_mask), torch.cuda.FloatTensor(x_descriptors))
            MAE = F.l1_loss(mol_prediction, torch.Tensor(y_val).view(-1,1), reduction='none')        
            MSE = F.mse_loss(mol_prediction, torch.Tensor(y_val).view(-1,1), reduction='none')
        
            test_MAE_list.extend(MAE.data.squeeze().cpu().numpy())
            test_MSE_list.extend(MSE.data.squeeze().cpu().numpy())
        return np.array(test_MAE_list).mean(), np.array(test_MSE_list).mean()


    def train_and_save():

        task_name = 'Task'
        tasks = ['Property']

        raw_filename = "data/dataset.csv"
        feature_filename = raw_filename.replace('.csv','.pickle')
        filename = raw_filename.replace('.csv','')
        prefix_filename = raw_filename.split('/')[-1].replace('.csv','')
        smiles_tasks_df = pd.read_csv(raw_filename, names = ["Property", "smiles"])
        smilesList = smiles_tasks_df.smiles.values
        atom_num_dist = []
        remained_smiles = []
        canonical_smiles_list = []
        for smiles in smilesList:
            try:        
                mol = Chem.MolFromSmiles(smiles)
                atom_num_dist.append(len(mol.GetAtoms()))
                remained_smiles.append(smiles)
                canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
            except:
                print(smiles)
                pass
        print("number of successfully processed smiles: ", len(remained_smiles))
        smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["smiles"].isin(remained_smiles)]
        smiles_tasks_df['cano_smiles'] =canonical_smiles_list

        random_seed = 68
        start_time = str(time.ctime()).replace(':','-').replace(' ','_')

        batch_size = 50#200
        epochs = 1#800

        p_dropout= 0.03
        fingerprint_dim = 200

        weight_decay = 4.3
        learning_rate = 4
        radius = 2
        T = 1
        per_task_output_units_num = 1 # for regression model
        output_units_num = len(tasks) * per_task_output_units_num

        if os.path.isfile(feature_filename):
            feature_dicts = pickle.load(open(feature_filename, "rb" ))
        else:
            feature_dicts = save_smiles_dicts(smilesList,filename)
        # feature_dicts = get_smiles_dicts(smilesList)
        remained_df = smiles_tasks_df[smiles_tasks_df["cano_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
        uncovered_df = smiles_tasks_df.drop(remained_df.index)

        test_df = remained_df.sample(frac=0.2,random_state=random_seed)
        train_df = remained_df.drop(test_df.index)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list, x_descriptors = get_smiles_array([canonical_smiles_list[0]],feature_dicts)
        num_atom_features = x_atom.shape[-1]
        num_bond_features = x_bonds.shape[-1]
        loss_function = nn.MSELoss()
        model = Descriptor(radius, T, num_atom_features, num_bond_features,
                fingerprint_dim, output_units_num, p_dropout)
        model.cuda()

        # optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
        optimizer = optim.Adam(model.parameters(), 10**-learning_rate, weight_decay=10**-weight_decay)
        # optimizer = optim.SGD(model.parameters(), 10**-learning_rate, weight_decay=10**-weight_decay)


        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])


        best_param ={}
        best_param["train_epoch"] = 0
        best_param["test_epoch"] = 0
        best_param["train_MSE"] = 9e8
        best_param["test_MSE"] = 9e8

        for epoch in range(epochs):#range(800):
            train_MAE, train_MSE = eval(model, train_df)
            test_MAE, test_MSE = eval(model, test_df)
            if train_MSE < best_param["train_MSE"]:
                best_param["train_epoch"] = epoch
                best_param["train_MSE"] = train_MSE
            if test_MSE < best_param["test_MSE"]:
                best_param["test_epoch"] = epoch
                best_param["test_MSE"] = test_MSE
                if test_MSE < 1.1:
                     torch.save(model, 'saved_models/model_'+prefix_filename+'_'+start_time+'_'+str(epoch)+'.pt')
            if (epoch - best_param["train_epoch"] >2) and (epoch - best_param["test_epoch"] >18):        
                break
            print(epoch, train_MSE, test_MSE)
    
            train(model, train_df, optimizer, loss_function)



        # evaluate model
        best_model = torch.load('saved/model_'+prefix_filename+'_'+start_time+'_'+str(best_param["test_epoch"])+'.pt')     

        best_model_dict = best_model.state_dict()
        best_model_wts = copy.deepcopy(best_model_dict)

        model.load_state_dict(best_model_wts)
        (best_model.align[0].weight == model.align[0].weight).all()
        test_MAE, test_MSE = eval(model, test_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type = int, default = 200)
    parser.add_argument('-dropout', type = float, default =  0.03)
    parser.add_argument('-epochs', type = int, default = 800)
    parser.add_argument('-desc_dim', type = int, default = 300)
    parser.add_argument('-hidden_desc_len', type = int, default = 200)
    parser.add_argument('-lr', type = float, default = 4.0)
    parser.add_argument('-radius', type = int, default = 2)
    parser.add_argument('-t', type = int, default = 1)
    parser.add_argument('-weight_decay', type = float, default = 4.3)
    args = parser.parse_args()

    experiment = Experiment(args)
    experiment.train_and_save()