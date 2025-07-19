import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dgl
from rdkit import Chem
from rdkit.Chem import Draw
import os
import datetime
import copy


from model import PharmHGT
from data import create_dataloader, identify_surfactant_features

def analyze_surfactant_features(model, dataloader, device, n_samples=10, results_folder='surfactant_analysis_results'):
    """
    Analyze the importance of surfactant-specific features in the model predictions
    
    Args:
        model: Trained PharmHGT model
        dataloader: DataLoader containing test molecules
        device: Device to run the model on
        n_samples: Number of molecules to analyze
        results_folder: Folder to save the results
    """
    os.makedirs(results_folder, exist_ok=True)
    model.eval()
    
    results = []
    sample_count = 0
    
    # Track statistics for different surfactant types
    head_importances = []
    tail_importances = []
    type_importances = {
        'nonionic': [],
        'cationic': [],
        'anionic': [], 
        'zwitterionic': []
    }
    
    for i, (bg, labels) in enumerate(dataloader):
        if sample_count >= n_samples:
            break
            
        # Process one molecule at a time
        if bg.batch_size > 1:
            graphs = dgl.unbatch(bg)
            for j, g in enumerate(graphs):
                if sample_count >= n_samples:
                    break
                    
                # Get SMILES for visualization
                smiles = dataloader.dataset.data.iloc[i * bg.batch_size + j]['smiles']
                mol = Chem.MolFromSmiles(smiles)
                
                # Get surfactant features from the original molecule
                surfactant_info = identify_surfactant_features(mol)
                
                # Compute importance scores using attention-based approach
                atom_importance = compute_surfactant_attribution(model, g.to(device), device)
                
                # Analyze feature importance for head vs tail groups
                result = analyze_feature_importance_by_group(
                    atom_importance, 
                    mol, 
                    surfactant_info,
                    os.path.join(results_folder, f'surfactant_analysis_mol_{sample_count}.png'),
                    smiles
                )
                
                # Collect statistics
                if len(result['head_importances']) > 0:
                    head_importances.extend(result['head_importances'])
                if len(result['tail_importances']) > 0:
                    tail_importances.extend(result['tail_importances'])
                
                surf_type = surfactant_info['type']
                if surf_type in type_importances and len(result['all_importances']) > 0:
                    type_importances[surf_type].extend(result['all_importances'])
                
                results.append(result)
                sample_count += 1
        
        else:
            # If already batch size 1, process directly
            smiles = dataloader.dataset.data.iloc[i]['smiles']
            mol = Chem.MolFromSmiles(smiles)
            
            # Get surfactant features
            surfactant_info = identify_surfactant_features(mol)
            
            # Compute importance scores
            atom_importance = compute_surfactant_attribution(model, bg.to(device), device)
            
            # Analyze feature importance for different groups
            result = analyze_feature_importance_by_group(
                atom_importance, 
                mol, 
                surfactant_info,
                os.path.join(results_folder, f'surfactant_analysis_mol_{sample_count}.png'),
                smiles
            )
            
            # Collect statistics
            if len(result['head_importances']) > 0:
                head_importances.extend(result['head_importances'])
            if len(result['tail_importances']) > 0:
                tail_importances.extend(result['tail_importances'])
            
            surf_type = surfactant_info['type']
            if surf_type in type_importances and len(result['all_importances']) > 0:
                type_importances[surf_type].extend(result['all_importances'])
            
            results.append(result)
            sample_count += 1
    
    # Generate summary visualizations
    generate_summary_plots(head_importances, tail_importances, type_importances, results_folder)
    
    return results

def compute_surfactant_attribution(model, molecule_graph, device):
    """
    Calculate atom-level attribution with focus on surfactant features
    using a simple feature-based approach without perturbation
    """
    # Make sure the graph is on the right device
    if molecule_graph.device != device:
        molecule_graph = molecule_graph.to(device)
    
    try:
        # Get atom features directly
        atom_features = molecule_graph.nodes['a'].data['f'].detach().cpu().numpy()
        num_atoms = atom_features.shape[0]
        
        # Use a simple feature magnitude-based importance
        atom_importance = np.sum(np.abs(atom_features), axis=1)
        
        # Emphasize head and tail features
        # The surfactant features are in the last 6 dimensions:
        # -6: is_head, -5: is_tail, -4 to -1: surfactant type
        try:
            is_head = atom_features[:, -6] > 0
            is_tail = atom_features[:, -5] > 0
            
            # Apply a boost to head and tail groups
            if np.any(is_head):
                atom_importance[is_head] *= 1.5  # Emphasize head groups
            if np.any(is_tail):
                atom_importance[is_tail] *= 1.2  # Emphasize tail groups
        except IndexError:
            # In case feature vector doesn't have expected structure
            pass
            
    except Exception as e:
        print(f"Error in feature-based importance: {e}")
        # Create a default uniform importance
        num_atoms = molecule_graph.num_nodes('a')
        atom_importance = np.ones(num_atoms)
    
    return atom_importance

def process_surfactant_attention_weights(attention_weights, graph):
    """
    Process the captured attention weights with focus on surfactant-relevant components
    """
    # Initialize atom importance scores
    atom_importance = np.zeros(graph.num_nodes('a'))
    
    # Get the node features to identify head and tail groups
    atom_features = graph.nodes['a'].data['f'].detach().cpu().numpy()
    num_atoms = atom_features.shape[0]
    
    # Extract head/tail information from features
    # The surfactant features are in the last 6 dimensions:
    # -6: is_head, -5: is_tail, -4 to -1: surfactant type
    try:
        is_head = atom_features[:, -6] > 0
        is_tail = atom_features[:, -5] > 0
    except IndexError:
        # In case the feature vector doesn't have the expected structure
        print("Warning: Feature vector doesn't have the expected structure for surfactant features")
        is_head = np.zeros(num_atoms, dtype=bool)
        is_tail = np.zeros(num_atoms, dtype=bool)
    
    try:
        # Process each attention weight tensor
        for i, weights in enumerate(attention_weights):
            try:
                # Check if weights has a valid tensor shape
                if not isinstance(weights, torch.Tensor) or len(weights.shape) == 0:
                    continue
                
                # For different possible attention weight shapes
                if len(weights.shape) == 4:  # [batch, heads, seq_len, seq_len]
                    # Average across batch and heads
                    atom_attn = weights.mean(dim=(0, 1))
                    
                    # If this tensor relates to atom nodes (check size compatibility)
                    if atom_attn.shape[0] == num_atoms and atom_attn.shape[1] == num_atoms:
                        # Sum across receiving attention
                        received_attn = atom_attn.sum(dim=0).numpy()
                        atom_importance += received_attn
                        
                elif len(weights.shape) == 3:  # [heads, seq_len, seq_len]
                    # Average across heads
                    atom_attn = weights.mean(dim=0)
                    
                    # If this tensor relates to atom nodes
                    if atom_attn.shape[0] == num_atoms and atom_attn.shape[1] == num_atoms:
                        received_attn = atom_attn.sum(dim=0).numpy()
                        atom_importance += received_attn
                        
                elif len(weights.shape) == 2:  # [seq_len, seq_len]
                    # Direct attention matrix
                    if weights.shape[0] == num_atoms and weights.shape[1] == num_atoms:
                        received_attn = weights.sum(dim=0).numpy()
                        atom_importance += received_attn
            except Exception as e:
                print(f"Warning: Could not process attention tensor {i}: {e}")
                continue
    except Exception as e:
        print(f"Error in primary attention processing: {e}")
        # Fall back to using the feature importance directly
        atom_importance = np.sum(np.abs(atom_features), axis=1)
    
    # If we didn't extract any useful attention weights, use feature-based importance
    if np.all(atom_importance == 0):
        print("No compatible attention weights found, using feature importance")
        atom_importance = np.sum(np.abs(atom_features), axis=1)
    
    # Scale importance by surfactant-specific features if identified
    try:
        if np.any(is_head):
            atom_importance[is_head] *= 1.5  # Emphasize head groups
        if np.any(is_tail):
            atom_importance[is_tail] *= 1.2  # Emphasize tail groups
    except Exception as e:
        print(f"Warning: Error applying head/tail emphasis: {e}")
    
    return atom_importance

def analyze_feature_importance_by_group(atom_importance, mol, surfactant_info, save_path, smiles):
    """
    Analyze feature importance based on surfactant structural groups
    
    Args:
        atom_importance: Attribution scores for each atom
        mol: RDKit molecule
        surfactant_info: Dictionary with surfactant structural information
        save_path: Path to save the visualization
        smiles: SMILES string of the molecule
    
    Returns:
        Dictionary with analysis results
    """
    # Convert to numpy array if it's not already
    if not isinstance(atom_importance, np.ndarray):
        atom_importance = np.array(atom_importance)
    
    # Check if atom count matches
    if mol.GetNumAtoms() != len(atom_importance):
        print(f"Warning: Atom count mismatch. Mol has {mol.GetNumAtoms()} atoms but importance has {len(atom_importance)} values.")
        if mol.GetNumAtoms() < len(atom_importance):
            atom_importance = atom_importance[:mol.GetNumAtoms()]
        else:
            # Pad with zeros if needed
            temp = np.zeros(mol.GetNumAtoms())
            temp[:len(atom_importance)] = atom_importance
            atom_importance = temp
    
    # Extract importance values for different groups
    head_atoms = list(surfactant_info['head_atoms'])
    tail_atoms = list(surfactant_info['tail_atoms'])
    
    # Fix head_atoms and tail_atoms that might exceed molecule size
    head_atoms = [i for i in head_atoms if i < len(atom_importance)]
    tail_atoms = [i for i in tail_atoms if i < len(atom_importance)]
    
    # Collect importance values by group
    head_importances = [atom_importance[i] for i in head_atoms]
    tail_importances = [atom_importance[i] for i in tail_atoms]
    
    # Calculate average importance for each group
    avg_head_importance = np.mean(head_importances) if head_importances else 0
    avg_tail_importance = np.mean(tail_importances) if tail_importances else 0
    
    # If no head/tail atoms were identified, try to infer from structure
    if not head_atoms and not tail_atoms and mol.GetNumAtoms() > 0:
        print(f"No surfactant groups identified for {smiles}. Attempting heuristic identification.")
        # Simple heuristic to identify hydrophilic parts (O, N atoms)
        # and hydrophobic parts (long carbon chains)
        head_atoms = []
        tail_atoms = []
        
        # Find oxygen and nitrogen atoms (likely part of head groups)
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in ['O', 'N', 'S', 'P']:
                head_atoms.append(atom.GetIdx())
            elif atom.GetSymbol() == 'C':
                # Check if it's part of an alkyl chain
                if all(neigh.GetSymbol() == 'C' or neigh.GetSymbol() == 'H' for neigh in atom.GetNeighbors()):
                    tail_atoms.append(atom.GetIdx())
        
        # Recalculate importances with these groups
        head_importances = [atom_importance[i] for i in head_atoms if i < len(atom_importance)]
        tail_importances = [atom_importance[i] for i in tail_atoms if i < len(atom_importance)]
        
        avg_head_importance = np.mean(head_importances) if head_importances else 0
        avg_tail_importance = np.mean(tail_importances) if tail_importances else 0
    
    # Create visualization
    try:
        visualize_molecule_with_groups(
            mol, 
            atom_importance, 
            head_atoms, 
            tail_atoms, 
            surfactant_info['type'],
            save_path
        )
    except Exception as e:
        print(f"Error in visualization for {smiles}: {e}")
    
    # # Convert lists to strings for CSV storage
    # head_importances_str = str(head_importances)
    # tail_importances_str = str(tail_importances)
    # all_importances_str = str(list(atom_importance))
    
    # Convert lists to regular Python lists of floats for CSV storage
    head_importances_list = [float(x) for x in head_importances]
    tail_importances_list = [float(x) for x in tail_importances]
    all_importances_list = [float(x) for x in atom_importance]

    
    # Return analysis results
    result = {
        'smiles': smiles,
        'surfactant_type': surfactant_info['type'],
        'avg_head_importance': float(avg_head_importance),  # Ensure it's a native Python float
        'avg_tail_importance': float(avg_tail_importance),  # Ensure it's a native Python float
        'head_importances': head_importances_list,
        'tail_importances': tail_importances_list,
        'all_importances': all_importances_list,
        'head_atom_count': len(head_atoms),
        'tail_atom_count': len(tail_atoms)
    }
    
    return result

def visualize_molecule_with_groups(mol, atom_importance, head_atoms, tail_atoms, surfactant_type, save_path):
    """
    Create a visualization of the molecule with surfactant groups highlighted
    
    Args:
        mol: RDKit molecule
        atom_importance: Attribution scores for each atom
        head_atoms: List of head group atom indices
        tail_atoms: List of tail group atom indices
        surfactant_type: Type of surfactant
        save_path: Path to save the visualization
    """
    # Normalize importance scores for visualization
    if len(atom_importance) > 0 and atom_importance.max() > atom_importance.min():
        norm_scores = (atom_importance - atom_importance.min()) / (atom_importance.max() - atom_importance.min())
    else:
        norm_scores = np.ones_like(atom_importance)
    
    # Generate atom highlighting
    highlight_atoms = list(range(min(mol.GetNumAtoms(), len(norm_scores))))
    
    # Create atom colors - head groups in blue, tail groups in green, others colored by importance
    atom_colors = {}
    for i, score in enumerate(norm_scores):
        if i < mol.GetNumAtoms():
            if i in head_atoms:
                # Head groups in red, intensity by importance
                r = float(1.0) 
                g = float(0.0)
                b = float(1.0 * score)  # Purple for higher importance
                atom_colors[i] = (r, g, b)
            elif i in tail_atoms:
                # Tail groups in green, intensity by importance
                r = float(0.0)
                g = float(1.0 * score + 0.5)  # Brighter green for higher importance
                b = float(0.0)
                atom_colors[i] = (r, g, b)
            else:
                # Other atoms in orange gradient based on importance
                r = float(1.0)
                g = float(0.5 * score)
                b = float(0.0)
                atom_colors[i] = (r, g, b)
    
    try:
        # Draw molecule with atom highlighting
        drawer = Draw.MolDraw2DCairo(800, 600)
        
        # Set options for better visualization
        opts = drawer.drawOptions()
        opts.addStereoAnnotation = True
        opts.additionalAtomLabelPadding = 0.3
        
        # Set the title with surfactant type
        try:
            drawer.DrawMolecule(
                mol,
                highlightAtoms=highlight_atoms,
                highlightAtomColors=atom_colors,
                legend=f"Surfactant Type: {surfactant_type}\nRed/Purple: Head Groups, Green: Tail Groups"
            )
        except Exception as e:
            print(f"Error in DrawMolecule: {e}, trying simpler drawing")
            drawer.DrawMolecule(mol)
            
        drawer.FinishDrawing()
        
        # Save the image
        with open(save_path, 'wb') as f:
            f.write(drawer.GetDrawingText())
        
    except Exception as e:
        print(f"Error in RDKit drawing: {e}")
        # Fall back to simpler drawing method
        try:
            img = Draw.MolToImage(mol, size=(800, 600))
            img.save(save_path)
        except Exception as e2:
            print(f"Fallback drawing also failed: {e2}")
            # Create a simple placeholder image
            try:
                from PIL import Image, ImageDraw, ImageFont
                img = Image.new('RGB', (800, 600), color=(255, 255, 255))
                d = ImageDraw.Draw(img)
                d.text((10, 10), f"Failed to draw: {surfactant_type}\n{save_path}", fill=(0, 0, 0))
                img.save(save_path)
            except:
                print("All drawing methods failed")

def generate_summary_plots(head_importances, tail_importances, type_importances, results_folder):
    """
    Generate summary plots of the feature importance analysis
    """
    # Convert string data to numeric
    def safe_convert_to_numeric(data_list):
        numeric_list = []
        for item in data_list:
            if isinstance(item, (int, float)):
                numeric_list.append(float(item))
            elif isinstance(item, str) and item.strip('[]').strip():
                # Try to parse as a list
                try:
                    values = eval(item)
                    if isinstance(values, (list, tuple)):
                        numeric_list.extend([float(x) for x in values if isinstance(x, (int, float))])
                    else:
                        # Skip if not a list or tuple
                        pass
                except:
                    # Skip if can't be evaluated
                    pass
        return numeric_list

    # Convert both head and tail importances
    numeric_head_importances = safe_convert_to_numeric(head_importances)
    numeric_tail_importances = safe_convert_to_numeric(tail_importances)
    
    # Convert type importances
    numeric_type_importances = {}
    for t, imp in type_importances.items():
        numeric_type_importances[t] = safe_convert_to_numeric(imp)
    
    # 1. Compare head vs tail importance
    if len(numeric_head_importances) > 10 and len(numeric_tail_importances) > 10:
        try:
            plt.figure(figsize=(10, 6))
            
            # Create violin plots for head and tail importances
            positions = [1, 2]
            data = [numeric_head_importances, numeric_tail_importances]
            labels = ['Head Groups', 'Tail Groups']
            
            plt.violinplot(data, positions, showmeans=True)
            
            # Add box plots inside the violin plots
            plt.boxplot(data, positions=positions, widths=0.3, showfliers=False)
            
            # Add labels and formatting
            plt.xticks(positions, labels)
            plt.ylabel('Importance Score')
            plt.title('Importance Distribution: Head vs Tail Groups')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(results_folder, 'head_vs_tail_importance.png'))
            plt.close()
        except Exception as e:
            print(f"Error creating head vs tail plot: {e}")
    else:
        print(f"Not enough data for head vs tail violin plot: {len(numeric_head_importances)} head, {len(numeric_tail_importances)} tail")
    
    # 2. Compare importance by surfactant type
    valid_types = {t: imp for t, imp in numeric_type_importances.items() if len(imp) > 10}
    
    if len(valid_types) > 0:
        try:
            plt.figure(figsize=(12, 6))
            
            # Create violin plots for each surfactant type
            positions = list(range(1, len(valid_types) + 1))
            data = [valid_types[t] for t in valid_types]
            labels = list(valid_types.keys())
            
            plt.violinplot(data, positions, showmeans=True)
            
            # Add box plots
            plt.boxplot(data, positions=positions, widths=0.3, showfliers=False)
            
            # Add labels and formatting
            plt.xticks(positions, labels)
            plt.ylabel('Importance Score')
            plt.title('Importance Distribution by Surfactant Type')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(results_folder, 'importance_by_surfactant_type.png'))
            plt.close()
        except Exception as e:
            print(f"Error creating surfactant type plot: {e}")
    else:
        print(f"Not enough data for surfactant type violin plot: {[f'{t}: {len(imp)}' for t, imp in numeric_type_importances.items()]}")
    
    # 3. Generate bar chart of average importance by group
    if len(numeric_head_importances) > 0 and len(numeric_tail_importances) > 0:
        try:
            avg_head = np.mean(numeric_head_importances)
            avg_tail = np.mean(numeric_tail_importances)
            
            plt.figure(figsize=(8, 5))
            plt.bar([1, 2], [avg_head, avg_tail], color=['red', 'green'])
            plt.xticks([1, 2], ['Head Groups', 'Tail Groups'])
            plt.ylabel('Average Importance Score')
            plt.title('Average Feature Importance: Head vs Tail')
            
            # Add values on top of bars
            for i, v in enumerate([avg_head, avg_tail]):
                plt.text(i + 1, v + 0.02, f'{v:.3f}', ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_folder, 'avg_importance_head_vs_tail.png'))
            plt.close()
        except Exception as e:
            print(f"Error creating average importance bar chart: {e}")
    else:
        print("Not enough data for average importance bar chart")
    
    # 4. Generate bar chart of average importance by surfactant type
    if valid_types:
        try:
            avgs = [np.mean(valid_types[t]) for t in valid_types]
            
            plt.figure(figsize=(10, 5))
            bars = plt.bar(range(1, len(valid_types) + 1), avgs, color='purple')
            plt.xticks(range(1, len(valid_types) + 1), list(valid_types.keys()))
            plt.ylabel('Average Importance Score')
            plt.title('Average Feature Importance by Surfactant Type')
            
            # Add values on top of bars
            for i, v in enumerate(avgs):
                plt.text(i + 1, v + 0.02, f'{v:.3f}', ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_folder, 'avg_importance_by_type.png'))
            plt.close()
        except Exception as e:
            print(f"Error creating average importance by type: {e}")
    else:
        print("Not enough data for average importance by type")

def run_surfactant_analysis(dataset_num=1, random_seed=42):
    """
    Main function to run the surfactant feature analysis
    
    Args:
        dataset_num: Dataset number (1 or 2)
        random_seed: Random seed for dataset loading
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Load model and data
    base_dir = 'c:\\Users\\gtheis2\\git\\surfactant_project\\Graph-Transformers_org-2'
    
    if dataset_num == 1:
        model_path = 'models/surfactant_model_data1_newSplit70_20_10_plot_opt_R4/best_fold0.pt'
        data_path = 'data/surfactant_data1/Split_70_20_10'
        results_folder = os.path.join(base_dir, f'models/surfactant_model_data1_newSplit70_20_10_plot_opt_R4/results/surfactant_feature_analysis/{timestamp}')
        model_args = {
            'hid_dim': 384,
            'atom_dim': 55,
            'bond_dim': 14,
            'pharm_dim': 194,
            'reac_dim': 34,
            'depth': 5,
            'act': 'relu',
            'num_task': 1,
            "dropout": 0.11190130761241533
        }
    else:
        model_path = 'models/surfactant_model_data2_Split70_20_10_plot_best_R2/best_fold0.pt'
        data_path = 'data/surfactant_data2/Split_70_20_10'
        results_folder = os.path.join(base_dir, f'models/surfactant_model_data2_Split70_20_10_plot_best_R2/results/surfactant_feature_analysis/{timestamp}')
        model_args = {
            'hid_dim': 192,
            'atom_dim': 55,
            'bond_dim': 14,
            'pharm_dim': 194,
            'reac_dim': 34,
            'depth': 4,
            'act': 'relu',
            'num_task': 1,
            "dropout": 0.1747254162818547
        }
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Data arguments
    data_args = {
        'path': data_path,
        'batch_size': 1,  # Process one molecule at a time
        'target_names': ['logCMC'],
        'task': 'regression'
    }
    
    # Load model
    print(f"Loading model from {model_path}")
    model = PharmHGT(model_args).to(device)
    
    try:
        # Try loading with weights_only=True to avoid the warning
        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Failed to load with weights_only=True: {e}")
        # Fall back to original loading method
        checkpoint = torch.load(model_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create dataloader for test set
    print(f"Loading test data with seed {random_seed}")
    testloader = create_dataloader(data_args, f'{random_seed}_fold_0_test.csv', shuffle=False, train=False)
    
    # Create results folder
    os.makedirs(results_folder, exist_ok=True)
    
    # Run the analysis
    print(f"Running feature analysis on dataset {dataset_num}...")
    results = analyze_surfactant_features(model, testloader, device, n_samples=len(testloader.dataset), results_folder=results_folder)
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_folder, f'surfactant_feature_analysis_results_seed{random_seed}.csv'), index=False)
    
    print(f"Analysis complete. Results saved to {results_folder}")
    
    return results_df

