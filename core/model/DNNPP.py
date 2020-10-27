import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Descriptor(nn.Module):

    def __init__(self, radius, T, input_feature_dim, input_bond_dim,\
            descriptor_dim, output_units_num, p_dropout):
        super(Descriptor, self).__init__()

        self.atom_linear_layer = nn.Linear(input_feature_dim, descriptor_dim)
        self.neighbor_linear_layer = nn.Linear(input_feature_dim + input_bond_dim, descriptor_dim)
        self.GRUCell = nn.ModuleList([nn.GRUCell(descriptor_dim, descriptor_dim) for r in range(radius)])
        self.align = nn.ModuleList([nn.Linear(2*descriptor_dim,1) for r in range(radius)])
        self.attend = nn.ModuleList([nn.Linear(descriptor_dim, descriptor_dim) for r in range(radius)])

        self.mol_GRUCell = nn.GRUCell(descriptor_dim, descriptor_dim)
        self.mol_align = nn.Linear(2*descriptor_dim,1)
        self.mol_attend = nn.Linear(descriptor_dim, descriptor_dim)
        self.dropout = nn.Dropout(p = p_dropout)
        self.relu = nn.ReLU()
        
        self.fc_g1 = torch.nn.Linear(descriptor_dim, 128)
        
        self.sn1 = nn.Linear(200, 512)
        self.sn2 = nn.Linear(512, 1024)
        self.sn3 = nn.Linear(1024, 200)
        
        self.fc1 = nn.Linear(328, 1312)
        self.fc2 = nn.Linear(1312, 656)
        self.output = nn.Linear(656, output_units_num)

        self.radius = radius
        self.T = T


    def SmallNetwork(self, features):
            features = self.sn1(features) #ELU, SELU...
            features = F.relu(features)
            features = self.sn2(features)
            features = F.relu(features)
            features = self.sn3(features)

            return features
    
    
    def forward(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask, descriptors):
        
        desc = list(descriptors)
        
        atom_mask = atom_mask.unsqueeze(2)
        batch_size,mol_length,num_atom_feat = atom_list.size()
        atom_feature = F.leaky_relu(self.atom_linear_layer(atom_list))

        bond_neighbor = [bond_list[i][bond_degree_list[i]] for i in range(batch_size)]
        bond_neighbor = torch.stack(bond_neighbor, dim=0)
        atom_neighbor = [atom_list[i][atom_degree_list[i]] for i in range(batch_size)]
        atom_neighbor = torch.stack(atom_neighbor, dim=0)
        
        neighbor_feature = torch.cat([atom_neighbor, bond_neighbor],dim=-1)
        neighbor_feature = F.leaky_relu(self.neighbor_linear_layer(neighbor_feature))

        
        attend_mask = atom_degree_list.clone()
        attend_mask[attend_mask != mol_length-1] = 1
        attend_mask[attend_mask == mol_length-1] = 0
        attend_mask = attend_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        softmax_mask = atom_degree_list.clone()
        softmax_mask[softmax_mask != mol_length-1] = 0
        softmax_mask[softmax_mask == mol_length-1] = -9e8
        softmax_mask = softmax_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        batch_size, mol_length, max_neighbor_num, descriptor_dim = neighbor_feature.shape
        atom_feature_expand = atom_feature.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, descriptor_dim)
        feature_align = torch.cat([atom_feature_expand, neighbor_feature],dim=-1)
        
        align_score = F.leaky_relu(self.align[0](self.dropout(feature_align)))
        
        align_score = align_score + softmax_mask
        attention_weight = F.softmax(align_score,-2)
        
        attention_weight = attention_weight * attend_mask

        neighbor_feature_transform = self.attend[0](self.dropout(neighbor_feature))

        context = torch.sum(torch.mul(attention_weight,neighbor_feature_transform),-2)

        context = F.elu(context)
        context_reshape = context.view(batch_size*mol_length, descriptor_dim)
        atom_feature_reshape = atom_feature.view(batch_size*mol_length, descriptor_dim)
        atom_feature_reshape = self.GRUCell[0](context_reshape, atom_feature_reshape)
        atom_feature = atom_feature_reshape.view(batch_size, mol_length, descriptor_dim)

        activated_features = F.relu(atom_feature)

        for d in range(self.radius-1):
            neighbor_feature = [activated_features[i][atom_degree_list[i]] for i in range(batch_size)]
            
            neighbor_feature = torch.stack(neighbor_feature, dim=0)
            atom_feature_expand = activated_features.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, descriptor_dim)

            feature_align = torch.cat([atom_feature_expand, neighbor_feature],dim=-1)

            align_score = F.leaky_relu(self.align[d+1](self.dropout(feature_align)))

            align_score = align_score + softmax_mask
            attention_weight = F.softmax(align_score,-2)

            attention_weight = attention_weight * attend_mask

            neighbor_feature_transform = self.attend[d+1](self.dropout(neighbor_feature))

            context = torch.sum(torch.mul(attention_weight,neighbor_feature_transform),-2)

            context = F.elu(context)
            context_reshape = context.view(batch_size*mol_length, descriptor_dim)

            atom_feature_reshape = self.GRUCell[d+1](context_reshape, atom_feature_reshape)
            atom_feature = atom_feature_reshape.view(batch_size, mol_length, descriptor_dim)
            

            activated_features = F.relu(atom_feature)

        mol_feature = torch.sum(activated_features * atom_mask, dim=-2)
        

        activated_features_mol = F.relu(mol_feature)           
        
        mol_softmax_mask = atom_mask.clone()
        mol_softmax_mask[mol_softmax_mask == 0] = -9e8
        mol_softmax_mask[mol_softmax_mask == 1] = 0
        mol_softmax_mask = mol_softmax_mask.type(torch.cuda.FloatTensor)
        
        for t in range(self.T):
            
            mol_prediction_expand = activated_features_mol.unsqueeze(-2).expand(batch_size, mol_length, descriptor_dim)
            mol_align = torch.cat([mol_prediction_expand, activated_features], dim=-1)
            mol_align_score = F.leaky_relu(self.mol_align(mol_align))
            mol_align_score = mol_align_score + mol_softmax_mask
            mol_attention_weight = F.softmax(mol_align_score,-2)
            mol_attention_weight = mol_attention_weight * atom_mask

            activated_features_transform = self.mol_attend(self.dropout(activated_features))

            mol_context = torch.sum(torch.mul(mol_attention_weight,activated_features_transform),-2)

            mol_context = F.elu(mol_context)
            mol_feature = self.mol_GRUCell(mol_context, mol_feature)

            activated_features_mol = F.relu(mol_feature)           
        
        x = self.dropout(mol_feature)
        x = self.fc_g1(x)

        for i in range(len(desc)):
            desc[i] = torch.unsqueeze(desc[i], 0)
        desc = torch.cat(desc, 0)
        desc = self.SmallNetwork(desc)
        
        x_con_desc = torch.cat((x, desc), 1)
        mol_feature = self.fc1(x_con_desc)
        mol_feature = self.relu(mol_feature)
        mol_feature = self.dropout(mol_feature)
        mol_feature = self.fc2(mol_feature)
        mol_feature = self.relu(mol_feature)
        mol_feature = self.dropout(mol_feature)
        mol_prediction = self.output(mol_feature)
        
        return atom_feature, mol_prediction