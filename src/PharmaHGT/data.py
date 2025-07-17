import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from torch.utils.data import Dataset
import dgl
from dgl.dataloading import GraphDataLoader
from rdkit.Chem.BRICS import FindBRICSBonds, BreakBRICSBonds
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import MACCSkeys
from rdkit import RDConfig
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')  
import os
import random
import re


fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

def bond_features(bond: Chem.rdchem.Bond):
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    
    # # Debug print
    # print(f"Bond feature dimensions: {len(fbond)}")  
    # return fbond

    return fbond

def pharm_property_types_feats(mol,factory=factory): 
    types = [i.split('.')[1] for i in factory.GetFeatureDefs().keys()]
    feats = [i.GetType() for i in factory.GetFeaturesForMol(mol)]
    result = [0] * len(types)
    for i in range(len(types)):
        if types[i] in list(set(feats)):
            result[i] = 1
    return result

def GetBricsBonds(mol):
    brics_bonds = list()
    brics_bonds_rules = list()
    bonds_tmp = FindBRICSBonds(mol)
    bonds = [b for b in bonds_tmp]
    for item in bonds:# item[0] is bond, item[1] is brics type
        brics_bonds.append([int(item[0][0]), int(item[0][1])])
        brics_bonds_rules.append([[int(item[0][0]), int(item[0][1])], GetBricsBondFeature([item[1][0], item[1][1]])])
        brics_bonds.append([int(item[0][1]), int(item[0][0])])
        brics_bonds_rules.append([[int(item[0][1]), int(item[0][0])], GetBricsBondFeature([item[1][1], item[1][0]])])

    result = []
    for bond in mol.GetBonds():
        beginatom = bond.GetBeginAtomIdx()
        endatom = bond.GetEndAtomIdx()
        if [beginatom, endatom] in brics_bonds:
            result.append([bond.GetIdx(), beginatom, endatom])
            
    return result, brics_bonds_rules

def GetBricsBondFeature(action):
    result = []
    start_action_bond = int(action[0]) if (action[0] !='7a' and action[0] !='7b') else 7
    end_action_bond = int(action[1]) if (action[1] !='7a' and action[1] !='7b') else 7
    emb_0 = [0 for i in range(17)]
    emb_1 = [0 for i in range(17)]
    emb_0[start_action_bond] = 1
    emb_1[end_action_bond] = 1
    result = emb_0 + emb_1
    return result

def maccskeys_emb(mol):
    return list(MACCSkeys.GenMACCSKeys(mol))

def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx()+1) # aviod index 0
    return mol

def GetFragmentFeats(mol):
    break_bonds = [mol.GetBondBetweenAtoms(i[0][0],i[0][1]).GetIdx() for i in FindBRICSBonds(mol)]
    if break_bonds == []:
        tmp = mol
    else:
        tmp = Chem.FragmentOnBonds(mol,break_bonds,addDummies=False)
    frags_idx_lst = Chem.GetMolFrags(tmp)
    result_ap = {}
    result_p = {}
    pharm_id = 0
    for frag_idx in frags_idx_lst:
        for atom_id in frag_idx:
            result_ap[atom_id] = pharm_id
        try:
            mol_pharm = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(mol, frag_idx))
            emb_0 = maccskeys_emb(mol_pharm)
            emb_1 = pharm_property_types_feats(mol_pharm)
        except Exception:
            emb_0 = [0 for i in range(167)]
            emb_1 = [0 for i in range(27)]
            
        result_p[pharm_id] = emb_0 + emb_1

        pharm_id += 1
    return result_ap, result_p

ELEMENTS = [35, 6, 7, 8, 9, 15, 16, 17, 53, 11, 19, 20, 23, 30, 13]
ATOM_FEATURES = {
    'atomic_num': ELEMENTS,
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 0, 1, 2],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
    'is_surfactant_head': [0, 1],  # New feature for surfactant head groups
    'is_surfactant_tail': [0, 1],  # New feature for surfactant tail groups
    'surfactant_type': [0, 1, 2, 3]  # [nonionic, cationic, anionic, zwitterionic]
}

# def identify_surfactant_features(mol):
#     """
#     Identify surfactant-specific structural features
#     Returns: Dictionary with surfactant type and head/tail atoms
#     """
#     surfactant_features = {
#         'type': 'unknown',
#         'head_atoms': set(),
#         'tail_atoms': set()
#     }
    
#     # Identify ionic groups for surfactant type classification
#     for atom in mol.GetAtoms():
#         # Anionic groups (e.g., sulfates, sulfonates, carboxylates)
#         if atom.GetFormalCharge() < 0 or (atom.GetSymbol() in ['S', 'P'] and 
#             len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'O']) >= 3):
#             surfactant_features['type'] = 'anionic'
#             surfactant_features['head_atoms'].add(atom.GetIdx())
            
#         # Cationic groups (e.g., quaternary ammonium)
#         elif atom.GetFormalCharge() > 0 or (atom.GetSymbol() == 'N' and 
#             atom.GetDegree() == 4):
#             surfactant_features['type'] = 'cationic'
#             surfactant_features['head_atoms'].add(atom.GetIdx())
            
#     # Identify hydrocarbon tails
#     carbon_chains = find_carbon_chains(mol)
#     if carbon_chains:
#         surfactant_features['tail_atoms'].update(max(carbon_chains, key=len))
    
#     # Classify zwitterionic if both positive and negative charges present
#     if any(a.GetFormalCharge() > 0 for a in mol.GetAtoms()) and \
#        any(a.GetFormalCharge() < 0 for a in mol.GetAtoms()):
#         surfactant_features['type'] = 'zwitterionic'
        
#     # Classify nonionic if no charges and contains both polar and nonpolar regions
#     elif surfactant_features['type'] == 'unknown' and \
#          has_polar_nonpolar_regions(mol):
#         surfactant_features['type'] = 'nonionic'
    
#     return surfactant_features

def identify_surfactant_features(mol):
    """
    Identify surfactant-specific structural features with SMILES pattern-based classification
    Returns: Dictionary with surfactant type and head/tail atoms
    """
    surfactant_features = {
        'type': 'unknown',
        'head_atoms': set(),
        'tail_atoms': set()
    }
    
    # First use your existing SMILES-based classification to determine the type
    try:
        smiles = Chem.MolToSmiles(mol)
        surfactant_features['type'] = classify_surfactant(smiles)
    except Exception as e:
        print(f"Error in SMILES classification: {e}")
    
    # Identify head and tail groups based on molecular structure
    # First pass: identify ionic and polar groups for head atoms
    for atom in mol.GetAtoms():
        # Anionic groups
        if (atom.GetFormalCharge() < 0 or 
            (atom.GetSymbol() in ['S', 'P'] and 
             sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == 'O') >= 3) or
            (atom.GetSymbol() == 'C' and 
             sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == 'O') > 1)):
            surfactant_features['head_atoms'].add(atom.GetIdx())
            # Add connected atoms to head group as well
            for neighbor in atom.GetNeighbors():
                surfactant_features['head_atoms'].add(neighbor.GetIdx())
            
        # Cationic groups
        elif (atom.GetFormalCharge() > 0 or 
              (atom.GetSymbol() == 'N' and atom.GetDegree() >= 3) or
              atom.GetSymbol() in ['K', 'Na']):  # Include counter-ions
            surfactant_features['head_atoms'].add(atom.GetIdx())
            # Add connected atoms to head group as well
            for neighbor in atom.GetNeighbors():
                surfactant_features['head_atoms'].add(neighbor.GetIdx())
                
        # Other polar head group atoms
        elif atom.GetSymbol() in ['O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I']:
            surfactant_features['head_atoms'].add(atom.GetIdx())
    
    # Identify hydrocarbon tails
    carbon_chains = find_carbon_chains(mol)
    if carbon_chains:
        # Use the longest chain for the tail
        longest_chain = max(carbon_chains, key=len)
        surfactant_features['tail_atoms'].update(longest_chain)
    
    # If no type was determined by SMILES classification, use structural features
    if surfactant_features['type'] == 'unknown':
        if any(atom.GetFormalCharge() > 0 and atom.GetFormalCharge() < 0 for atom in mol.GetAtoms()):
            surfactant_features['type'] = 'zwitterionic'
        elif any(atom.GetFormalCharge() < 0 for atom in mol.GetAtoms()):
            surfactant_features['type'] = 'anionic'
        elif any(atom.GetFormalCharge() > 0 for atom in mol.GetAtoms()):
            surfactant_features['type'] = 'cationic'
        elif len(surfactant_features['head_atoms']) > 0 and len(surfactant_features['tail_atoms']) > 0:
            surfactant_features['type'] = 'nonionic'
    
    return surfactant_features

def classify_surfactant(smiles):
    """
    Enhanced classification of surfactants based on SMILES patterns
    """
    if isinstance(smiles, bytes):
        smiles = smiles.decode('utf-8')
   
    # Split into molecular components (separate counter ions)
    molecular_units = smiles.split('.')
   
    # Analyze main molecular unit (typically the largest one)
    main_molecule = max(molecular_units, key=len)
   
    # Anionic patterns - only look in main molecule
    anionic_patterns = [
        r'\[O-\]',            # Explicit negative oxygen
        r'\[S-\]',            # Explicit negative sulfur
        r'S\(=O\)\(=O\)\[O-\]',  # Sulfonate group
        r'C\(=O\)\[O-\]',     # Carboxylate group
        r'P\(=O\)\[O-\]',     # Phosphonate group
    ]
   
    # Cationic patterns - only look in main molecule
    cationic_patterns = [
        r'\[N\+\]',           # Explicit positive nitrogen
        r'\[NH\+\]',          # Protonated amine
        r'\[NH2\+\]',         # Protonated primary amine
        r'\[NH3\+\]',         # Protonated primary amine
        r'\[P\+\]',           # Explicit positive phosphorus
        r'\[S\+\]',           # Explicit positive sulfur
        r'\[n\+\]',           # Aromatic nitrogen cation
    ]
   
    # Check for charged groups only in main molecule
    has_anionic_main = any(re.search(pattern, main_molecule) for pattern in anionic_patterns)
    has_cationic_main = any(re.search(pattern, main_molecule) for pattern in cationic_patterns)
   
    # Check counter ions to determine type
    counter_ions = [unit for unit in molecular_units if unit != main_molecule]
    has_anionic_counter = any(ion in ['[Cl-]', '[Br-]', '[I-]', '[O-]', '[SO4-]', '[NO3-]'] for ion in counter_ions)
    has_cationic_counter = any(ion in ['[Na+]', '[K+]', '[Li+]', '[NH4+]'] for ion in counter_ions)
   
    # Classification logic
    if has_anionic_main and has_cationic_main:
        return 'zwitterionic'  # Both charges in main molecule = zwitterionic
    elif has_cationic_main or (counter_ions and has_anionic_counter):  # Including aromatic nitrogen cations with Cl- counter ion
        return 'cationic'
    elif has_anionic_main or (counter_ions and has_cationic_counter):
        return 'anionic'
    else:
        return 'nonionic'

def find_carbon_chains(mol):
    """Find continuous carbon chains in molecule"""
    chains = []
    visited = set()
    
    def dfs_carbon_chain(atom, current_chain):
        if atom.GetIdx() in visited or atom.GetSymbol() != 'C':
            chains.append(current_chain.copy())
            return
        
        visited.add(atom.GetIdx())
        current_chain.append(atom.GetIdx())
        
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == 'C':
                dfs_carbon_chain(neighbor, current_chain)
    
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C' and atom.GetIdx() not in visited:
            dfs_carbon_chain(atom, [])
            
    return [chain for chain in chains if len(chain) >= 4]  # Min length for tail

def has_polar_nonpolar_regions(mol):
    """Check if molecule has distinct polar and nonpolar regions"""
    polar_atoms = {'O', 'N', 'S', 'P'}
    has_polar = any(a.GetSymbol() in polar_atoms for a in mol.GetAtoms())
    has_nonpolar = any(len(chain) >= 4 for chain in find_carbon_chains(mol))
    return has_polar and has_nonpolar


def onek_encoding_unk(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

def atom_features(atom: Chem.rdchem.Atom, surfactant_info=None):
    if surfactant_info is None:
        surfactant_info = {'head_atoms': set(), 'tail_atoms': set(), 'type': 'unknown'}
        
    features = onek_encoding_unk(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [1 if atom.GetIdx() in surfactant_info['head_atoms'] else 0] + \
           [1 if atom.GetIdx() in surfactant_info['tail_atoms'] else 0] + \
           onek_encoding_unk(
               {'unknown': 0, 'nonionic': 0, 'cationic': 1, 'anionic': 2, 'zwitterionic': 3}[surfactant_info['type']], 
               list(range(4))
           ) + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    
    # # Print feature dimensions for debugging
    # print(f"Feature dimensions breakdown:")
    # print(f"Atomic number: {len(ATOM_FEATURES['atomic_num']) + 1}")
    # print(f"Degree: {len(ATOM_FEATURES['degree']) + 1}")
    # print(f"Formal charge: {len(ATOM_FEATURES['formal_charge']) + 1}")
    # print(f"Chiral tag: {len(ATOM_FEATURES['chiral_tag']) + 1}")
    # print(f"Num Hs: {len(ATOM_FEATURES['num_Hs']) + 1}")
    # print(f"Hybridization: {len(ATOM_FEATURES['hybridization']) + 1}")
    # print(f"Aromatic: 1")
    # print(f"Head group: 1")
    # print(f"Tail group: 1")
    # print(f"Surfactant type: 5")
    # print(f"Mass: 1")
    # print(f"Total features: {len(features)}")

    return features


def Mol2HeteroGraph(mol):

    surfactant_info = identify_surfactant_features(mol)
    
    # build graphs
    edge_types = [('a','b','a'),('p','r','p'),('a','j','p'), ('p','j','a')]

    edges = {k:[] for k in edge_types}
    # if mol.GetNumAtoms() == 1:
    #     g = dgl.heterograph(edges, num_nodes_dict={'a':1,'p':1})
    # else:
    result_ap, result_p = GetFragmentFeats(mol)
    reac_idx, bbr = GetBricsBonds(mol)

    # Add surfactant-specific features to atoms
    f_atom = []
    for idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(idx)
        f_atom.append(atom_features(atom, surfactant_info))

    for bond in mol.GetBonds(): 
        edges[('a','b','a')].append([bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()])
        edges[('a','b','a')].append([bond.GetEndAtomIdx(),bond.GetBeginAtomIdx()])

    for r in reac_idx:
        begin = r[1]
        end = r[2]
        edges[('p','r','p')].append([result_ap[begin],result_ap[end]])
        edges[('p','r','p')].append([result_ap[end],result_ap[begin]])

    for k,v in result_ap.items():
        edges[('a','j','p')].append([k,v])
        edges[('p','j','a')].append([v,k])

    g = dgl.heterograph(edges)
    
    f_atom = []
    for idx in g.nodes('a'):
        atom = mol.GetAtomWithIdx(idx.item())
        f_atom.append(atom_features(atom))
    f_atom = torch.FloatTensor(f_atom)
    g.nodes['a'].data['f'] = f_atom
    dim_atom = len(f_atom[0])

    f_pharm = []
    for k,v in result_p.items():
        f_pharm.append(v)
    g.nodes['p'].data['f'] = torch.FloatTensor(f_pharm)
    dim_pharm = len(f_pharm[0])
    
    
    dim_atom_padding = g.nodes['a'].data['f'].size()[0]
    dim_pharm_padding = g.nodes['p'].data['f'].size()[0]

    g.nodes['a'].data['f_junc'] = torch.cat([g.nodes['a'].data['f'], torch.zeros(dim_atom_padding, dim_pharm)], 1)
    g.nodes['p'].data['f_junc'] = torch.cat([torch.zeros(dim_pharm_padding, dim_atom), g.nodes['p'].data['f']], 1)
    
    # features of edges

    f_bond = []
    src,dst = g.edges(etype=('a','b','a'))
    for i in range(g.num_edges(etype=('a','b','a'))):
        f_bond.append(bond_features(mol.GetBondBetweenAtoms(src[i].item(),dst[i].item())))
    g.edges[('a','b','a')].data['x'] = torch.FloatTensor(f_bond)

    f_reac = []
    src, dst = g.edges(etype=('p','r','p'))
    for idx in range(g.num_edges(etype=('p','r','p'))):
        p0_g = src[idx].item()
        p1_g = dst[idx].item()
        for i in bbr:
            p0 = result_ap[i[0][0]]
            p1 = result_ap[i[0][1]]
            if p0_g == p0 and p1_g == p1:
                f_reac.append(i[1])
    g.edges[('p','r','p')].data['x'] = torch.FloatTensor(f_reac)

    return g



class MolGraphSet(Dataset):
    def __init__(self,df,target,log=print):
        self.data = df
        self.mols = []
        self.labels = []
        self.graphs = []
        for i,row in df.iterrows():
            smi = row['smiles']
            label = row[target].values.astype(float)
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    log('invalid',smi)
                else:
                    g = Mol2HeteroGraph(mol)
                    if g.num_nodes('a') == 0:
                        log('no edge in graph',smi)
                    else:
                        self.mols.append(mol)
                        self.graphs.append(g)
                        self.labels.append(label)
            except Exception as e:
                log(e,'invalid',smi)
                
    def __len__(self):
        return len(self.mols)
    
    def __getitem__(self,idx):
        
        return self.graphs[idx],self.labels[idx]
    
def create_dataloader(args,filename,shuffle=True,train=True):
    dataset = MolGraphSet(pd.read_csv(os.path.join(args['path'],filename)),args['target_names'])
    if train:
        batch_size = args['batch_size']
    else:
        batch_size = min(4200,len(dataset))
    
    dataloader = GraphDataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
    
    return dataloader
    

    
def random_split(load_path,save_dir,num_fold=5,sizes = [0.7,0.1,0.2],seed=0):
    df = pd.read_csv(load_path)
    n = len(df)
    os.makedirs(save_dir,exist_ok=True)
    # Set seeds for all sources of randomness
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for fold in range(num_fold):

        df = df.loc[torch.randperm(n)].reset_index(drop=True)
        train_size = int(sizes[0] * n)
        train_val_size = int((sizes[0] + sizes[1]) * n)
        train = df[:train_size]
        val = df[train_size:train_val_size]
        test = df[train_val_size:]
        train.to_csv(os.path.join(save_dir)+f'{seed}_fold_{fold}_train.csv',index=False)
        val.to_csv(os.path.join(save_dir)+f'{seed}_fold_{fold}_valid.csv',index=False)
        test.to_csv(os.path.join(save_dir)+f'{seed}_fold_{fold}_test.csv',index=False)



if __name__=='__main__':
    for seed in [2022]:
        random_split('data_index/esol.csv','data_index/esol/',seed=seed)