"""
Ablation Study Script for Surfactant CMC Model
This version is designed to be easy to run in a Jupyter notebook or as a standalone script.
@author: Gabi107
@date: 2025-11-04
"""

import os
import numpy as np
import pandas as pd
import torch
import dgl
from rdkit import Chem
from tqdm import tqdm
import json
import copy

# Import your model and data modules
from model import PharmHGT as Model
from data import Mol2HeteroGraph, identify_surfactant_features


import matplotlib as mpl
import matplotlib.pyplot as plt

# Global style and resolution settings
mpl.rcParams['figure.dpi'] = 600           # Higher resolution for export
mpl.rcParams['savefig.dpi'] = 600          # High-res when saving figures
mpl.rcParams['lines.linewidth'] = 1.25
mpl.rcParams['lines.markersize'] = 4
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
plt.rcParams.update({
    "text.usetex": False,         # Keep False unless LaTeX is installed
    "font.family": "serif",
    "font.serif": ["Times New Roman"]
})

class SimpleAblationStudy:
    """Simplified ablation study for quick experiments."""
    
    def __init__(self, model_path, config_path, save_dir=None):
        """
        Initialize with model and config paths.
        
        Args:
            model_path: Path to saved model (e.g., 'best_fold0.pt')
            config_path: Path to config JSON file
            save_dir: folder where figures/results will be saved
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # where to save
        if save_dir is None:
            save_dir = "./ablation_results_plots"
        else:
            save_dir = os.path.abspath(f'{save_dir}/ablation_results_plots')
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.model_args = self.config['model']
        
        # Load model
        self.model = Model(self.model_args).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            self.model.load_state_dict(checkpoint)
            print("Model loaded successfully")
        
        self.model.eval()
    
    # def ablate_molecule(self, smiles, ablation_type='none'):
    #     """
    #     Perform ablation on a single molecule.
        
    #     Args:
    #         smiles: SMILES string of the molecule
    #         ablation_type: Type of ablation
    #             - 'none': No ablation (baseline)
    #             - 'remove_head': Remove head group features
    #             - 'remove_tail': Remove tail group features
    #             - 'remove_both': Remove both head and tail features
    #             - 'zero_head': Zero out head atom features completely
    #             - 'zero_tail': Zero out tail atom features completely
        
    #     Returns:
    #         Prediction value (logCMC)
    #     """
    #     mol = Chem.MolFromSmiles(smiles)
    #     if mol is None:
    #         return None
        
    #     # Create graph
    #     g = Mol2HeteroGraph(mol)
        
    #     # Apply ablation
    #     if ablation_type != 'none':
    #         atom_features = g.nodes['a'].data['f'].clone()
    #         feature_dim = atom_features.shape[1]
            
    #         # Feature indices for head and tail indicators
    #         head_idx = feature_dim - 6
    #         tail_idx = feature_dim - 5
            
    #         if ablation_type == 'remove_head':
    #             # Remove head indicator
    #             atom_features[:, head_idx] = 0
    #         elif ablation_type == 'remove_tail':
    #             # Remove tail indicator
    #             atom_features[:, tail_idx] = 0
    #         elif ablation_type == 'remove_both':
    #             # Remove both indicators
    #             atom_features[:, head_idx] = 0
    #             atom_features[:, tail_idx] = 0
    #         elif ablation_type == 'zero_head':
    #             # Zero out all features for head atoms
    #             head_mask = (atom_features[:, head_idx] == 1)
    #             atom_features[head_mask] = 0
    #         elif ablation_type == 'zero_tail':
    #             # Zero out all features for tail atoms
    #             tail_mask = (atom_features[:, tail_idx] == 1)
    #             atom_features[tail_mask] = 0
            
    #         # Update graph
    #         g.nodes['a'].data['f'] = atom_features
            
    #         # Update junction features if present
    #         if 'f_junc' in g.nodes['a'].data:
    #             dim_atom = atom_features.shape[1]
    #             g.nodes['a'].data['f_junc'][:, :dim_atom] = atom_features
        
    #     # Make prediction
    #     with torch.no_grad():
    #         bg = dgl.batch([g]).to(self.device)
    #         pred = self.model(bg).cpu().numpy()[0, 0]
        
    #     return pred

    def _guess_indicator_indices(self, atom_features, head_mask, tail_mask):
        """
        Heuristic to find 'is_head' / 'is_tail' indicator columns.
        Returns (head_idx, tail_idx) or (None, None) if not found.
        Strategy:
          - Column is mostly binary (0/1).
          - Values align with head_mask / tail_mask significantly better than random.
        """

        X = atom_features.detach().cpu().numpy()
        H = head_mask.detach().cpu().numpy().astype(int)
        T = tail_mask.detach().cpu().numpy().astype(int)

        n_atoms, n_feats = X.shape
        head_idx = None
        tail_idx = None
    
        def is_binary(col):
            u = np.unique(col)
            return np.all(np.isin(u, [0., 1.])) or (u.size <= 3 and np.all((u >= -1e-6) & (u <= 1+1e-6)))
    
        # simple alignment score: accuracy vs mask
        def align_score(col, mask):
            col_bin = (col > 0.5).astype(int)
            return (col_bin == mask).mean()

        best_h = 0.0
        best_t = 0.0
        for j in range(n_feats):
            col = X[:, j]
            if not is_binary(col): 
                continue
            s_h = align_score(col, H)
            s_t = align_score(col, T)
            if s_h > 0.9 and s_h > best_h:  # strong alignment
                best_h = s_h
                head_idx = j
            if s_t > 0.9 and s_t > best_t:
                best_t = s_t
                tail_idx = j
    
        return head_idx, tail_idx
    
    def _get_head_tail_masks(self, mol, g):
        """
        Build boolean masks for head/tail atoms using identify_surfactant_features.
        Assumes DGL atom node order matches RDKit atom indices (usual case).
        """
        surf = identify_surfactant_features(mol)
        head_ids = set(surf.get('head_atoms', []))
        tail_ids = set(surf.get('tail_atoms', []))

        num_atoms = g.nodes['a'].data['f'].shape[0]
        device = g.nodes['a'].data['f'].device

        head_mask = torch.tensor([i in head_ids for i in range(num_atoms)],
                             dtype=torch.bool, device=device)
        tail_mask = torch.tensor([i in tail_ids for i in range(num_atoms)],
                             dtype=torch.bool, device=device)
        return head_mask, tail_mask, surf



    def ablate_molecule(self, smiles, ablation_type='none'):
        """
        Robust ablation that uses atom indices (not hardcoded feature columns).
        If indicator columns can be detected, 'remove_*' will only zero those;
        otherwise 'remove_*' falls back to 'zero_*'.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Build graph
        g = Mol2HeteroGraph(mol)
        atom_features = g.nodes['a'].data['f'].clone()
        head_mask, tail_mask, _ = self._get_head_tail_masks(mol, g)

        # Early exit: baseline
        if ablation_type == 'none':
            with torch.no_grad():
                bg = dgl.batch([g]).to(self.device)
                pred = self.model(bg).cpu().numpy()[0, 0]
            return pred

        # Try to find indicator columns (optional; best-effort)
        head_idx, tail_idx = self._guess_indicator_indices(atom_features, head_mask, tail_mask)

        def zero_rows(mask):
            atom_features[mask] = 0
            if 'f_junc' in g.nodes['a'].data:
                # Keep junction feature's atom slice in sync if present
                dim_atom = atom_features.shape[1]
                g.nodes['a'].data['f_junc'][mask, :dim_atom] = 0

        def zero_indicator(mask, idx):
            # If we can't find the indicator, fall back to zeroing full rows
            if idx is None:
                zero_rows(mask)
            else:
                atom_features[mask, idx] = 0
                if 'f_junc' in g.nodes['a'].data:
                    dim_atom = atom_features.shape[1]
                    g.nodes['a'].data['f_junc'][mask, idx] = 0  # same column

        # Apply requested ablation
        if ablation_type == 'zero_head':
            zero_rows(head_mask)
        elif ablation_type == 'zero_tail':
            zero_rows(tail_mask)
        elif ablation_type == 'remove_head':
            zero_indicator(head_mask, head_idx)
        elif ablation_type == 'remove_tail':
            zero_indicator(tail_mask, tail_idx)
        elif ablation_type == 'remove_both':
            if head_idx is None or tail_idx is None:
                # Fall back to stronger neutralization if we can't find both indicators
                zero_rows(head_mask | tail_mask)
            else:
                atom_features[head_mask, head_idx] = 0
                atom_features[tail_mask, tail_idx] = 0
                if 'f_junc' in g.nodes['a'].data:
                    dim_atom = atom_features.shape[1]
                    g.nodes['a'].data['f_junc'][head_mask, head_idx] = 0
                    g.nodes['a'].data['f_junc'][tail_mask, tail_idx] = 0
        else:
            raise ValueError(f"Unknown ablation_type: {ablation_type}")
    
        # Write back and predict
        g.nodes['a'].data['f'] = atom_features
    
        with torch.no_grad():
            bg = dgl.batch([g]).to(self.device)
            pred = self.model(bg).cpu().numpy()[0, 0]
    
        return pred

    
    def analyze_molecule(self, smiles, molecule_name=None):
        """
        Perform complete ablation analysis on a single molecule.
        
        Args:
            smiles: SMILES string
            molecule_name: Optional name for the molecule
        
        Returns:
            Dictionary with all ablation results
        """
        if molecule_name is None:
            molecule_name = smiles[:20] + "..." if len(smiles) > 20 else smiles
        
        print(f"\nAnalyzing: {molecule_name}")
        print("=" * 50)
        
        # Get surfactant info
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("Invalid SMILES!")
            return None
        
        g = Mol2HeteroGraph(mol)
        head_mask, tail_mask, surf_info = self._get_head_tail_masks(mol, g)

        print(f"Surfactant type: {surf_info.get('type', 'unknown')}")
        print(f"Head atoms: {len(surf_info.get('head_atoms', []))} / {mol.GetNumAtoms()}")
        print(f"Tail atoms: {len(surf_info.get('tail_atoms', []))} / {mol.GetNumAtoms()}")
        
        # surf_info = identify_surfactant_features(mol)
        # print(f"Surfactant type: {surf_info['type']}")
        # print(f"Head atoms: {len(surf_info['head_atoms'])} / {mol.GetNumAtoms()}")
        # print(f"Tail atoms: {len(surf_info['tail_atoms'])} / {mol.GetNumAtoms()}")
        
        # Run ablations
        results = {}
        ablation_types = ['none', 'remove_head', 'remove_tail', 'remove_both', 'zero_head', 'zero_tail']
        
        print("\nPredictions:")
        for abl_type in ablation_types:
            pred = self.ablate_molecule(smiles, abl_type)
            results[abl_type] = pred
            print(f"  {abl_type:15s}: {pred:.3f}")
        
        # Calculate impacts
        baseline = results['none']
        print("\nImpact Analysis (ΔlogCMC):")
        for abl_type in ablation_types[1:]:  # Skip 'none'
            delta = results[abl_type] - baseline
            pct = (delta / abs(baseline) * 100) if baseline != 0 else 0
            print(f"  {abl_type:15s}: {delta:+.3f} ({pct:+.1f}%)")
        
        return results
    
    def batch_analysis(self, csv_path, num_samples=10):
        """
        Analyze multiple molecules from a CSV file.
        
        Args:
            csv_path: Path to CSV with 'smiles' column
            num_samples: Number of molecules to analyze
        
        Returns:
            DataFrame with results
        """
        df = pd.read_csv(csv_path)
        
        if 'smiles' not in df.columns:
            print("Error: CSV must have 'smiles' column")
            return None
        
        # Sample molecules
        if num_samples and num_samples < len(df):
            df = df.sample(n=num_samples, random_state=42)
        
        results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing molecules"):
            smiles = row['smiles']
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                # Get predictions for all ablation types
                preds = {}
                for abl_type in ['none','remove_head','remove_tail','remove_both','zero_head','zero_tail']:
                    preds[abl_type] = self.ablate_molecule(smiles, abl_type)
                
                # Calculate impacts
                baseline = preds['none']
                
                result = {
                    'smiles': smiles,
                    'baseline_prediction': baseline,
                    'head_impact': preds['remove_head'] - baseline,
                    'tail_impact': preds['remove_tail'] - baseline,
                    'both_impact': preds['remove_both'] - baseline,
                    'head_importance': abs(preds['remove_head'] - baseline),
                    'tail_importance': abs(preds['remove_tail'] - baseline),
                    # (optional) strong ablation magnitudes:
                    'zero_head_impact': preds['zero_head'] - baseline,
                    'zero_tail_impact': preds['zero_tail'] - baseline,
                }
                
                # Add actual CMC if available
                if 'logCMC' in row:
                    result['actual_logCMC'] = row['logCMC']
                    result['baseline_error'] = baseline - row['logCMC']
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing molecule {idx}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        
        # Print summary statistics
        print("\n" + "="*60)
        print("ABLATION STUDY SUMMARY")
        print("="*60)
        print(f"Analyzed {len(results_df)} molecules\n")
        
        print("Average Impacts (ΔlogCMC):")
        print(f"  Head removal: {results_df['head_impact'].mean():.3f} ± {results_df['head_impact'].std():.3f}")
        print(f"  Tail removal: {results_df['tail_impact'].mean():.3f} ± {results_df['tail_impact'].std():.3f}")
        print(f"  Both removal: {results_df['both_impact'].mean():.3f} ± {results_df['both_impact'].std():.3f}")
        
        print("\nAverage Importance (|ΔlogCMC|):")
        print(f"  Head: {results_df['head_importance'].mean():.3f}")
        print(f"  Tail: {results_df['tail_importance'].mean():.3f}")
        
        # Create visualization
        self.plot_results(results_df)
        
        return results_df
    
    def plot_results(self, results_df):
        """Create visualization of ablation results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
                
        # 1. Distribution of impacts
        ax = axes[0, 0]
        data = [results_df['head_impact'], results_df['tail_impact'], results_df['both_impact']]
        bp = ax.boxplot(data, labels=['Head', 'Tail', 'Both'])
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_ylabel('Change in logCMC')
        ax.set_title('Distribution of Ablation Impacts')
        ax.grid(True, alpha=0.3)
        
        # 2. Head vs Tail importance
        ax = axes[0, 1]
        ax.scatter(results_df['head_importance'], results_df['tail_importance'], 
                  alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Head Importance (|ΔlogCMC|)')
        ax.set_ylabel('Tail Importance (|ΔlogCMC|)')
        ax.set_title('Head vs Tail Feature Importance')
        
        # Add diagonal line
        max_val = max(results_df['head_importance'].max(), results_df['tail_importance'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # 3. Average importance comparison
        ax = axes[1, 0]
        categories = ['Head', 'Tail', 'Both']
        values = [
            results_df['head_importance'].mean(),
            results_df['tail_importance'].mean(),
            results_df['both_impact'].abs().mean()
        ]
        errors = [
            results_df['head_importance'].std(),
            results_df['tail_importance'].std(),
            results_df['both_impact'].abs().std()
        ]
        
        bars = ax.bar(categories, values, yerr=errors, capsize=5, 
                      color=['skyblue', 'lightcoral', 'lightgreen'])
        ax.set_ylabel('Average |ΔlogCMC|')
        ax.set_title('Average Feature Importance')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Correlation between head and tail impacts
        ax = axes[1, 1]
        ax.scatter(results_df['head_impact'], results_df['tail_impact'], 
                  alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Head Impact (ΔlogCMC)')
        ax.set_ylabel('Tail Impact (ΔlogCMC)')
        ax.set_title('Head vs Tail Impact Correlation')
        
        # Add quadrant labels
        ax.text(0.05, 0.95, 'Head↓ Tail↑', transform=ax.transAxes, fontsize=8, va='top')
        ax.text(0.95, 0.95, 'Head↑ Tail↑', transform=ax.transAxes, fontsize=8, va='top', ha='right')
        ax.text(0.05, 0.05, 'Head↓ Tail↓', transform=ax.transAxes, fontsize=8)
        ax.text(0.95, 0.05, 'Head↑ Tail↓', transform=ax.transAxes, fontsize=8, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Ablation Study Results: Head vs Tail Contributions', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save figure
        out_png = os.path.join(self.save_dir, "Ablation_Study_Head_Tail.png")
        out_pdf = os.path.join(self.save_dir, "Ablation_Study_Head_Tail.pdf")
        plt.savefig(out_png, dpi=600, bbox_inches='tight')
        plt.savefig(out_pdf, bbox_inches='tight')

        plt.show()
        
        
        return fig


# Example usage function
def run_ablation_example():
    """
    Example function showing how to use the ablation study.
    Modify paths according to your setup.
    """
    
    # Set your paths here
    MODEL_PATH = "path/to/your/best_fold0.pt"  # Update this
    CONFIG_PATH = "path/to/your/config.json"   # Update this
    DATA_PATH = "path/to/your/test_data.csv"   # Update this
    
    # Initialize ablation study
    print("Loading model...")
    ablation = SimpleAblationStudy(MODEL_PATH, CONFIG_PATH)
    
    # Example 1: Analyze a single molecule
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Molecule Analysis")
    print("="*60)
    
    # Example surfactant SMILES (sodium dodecyl sulfate)
    sds_smiles = "CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]"
    results = ablation.analyze_molecule(sds_smiles, "Sodium Dodecyl Sulfate (SDS)")
    
    # Example 2: Batch analysis
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Analysis")
    print("="*60)
    
    results_df = ablation.batch_analysis(DATA_PATH, num_samples=20)
    
    # Save results
    if results_df is not None:
        results_df.to_csv("ablation_results.csv", index=False)
        print("\nResults saved to ablation_results.csv")
    
    return ablation, results_df


if __name__ == "__main__":
    # Quick test with a simple molecule
    print("Quick test with example molecule...")
    
    # You need to update these paths
    model_path = input("Enter path to saved model (e.g., best_fold0.pt): ").strip()
    config_path = input("Enter path to config JSON: ").strip()
    
    if os.path.exists(model_path) and os.path.exists(config_path):
        ablation = SimpleAblationStudy(model_path, config_path)
        
        # Test with a simple surfactant
        test_smiles = "CCCCCCCCCCCC(=O)[O-]"  # Laurate ion
        print(f"\nTesting with laurate ion: {test_smiles}")
        result = ablation.analyze_molecule(test_smiles, "Laurate ion")
        
        # Optional: batch analysis
        data_path = input("\nEnter path to test data CSV (or press Enter to skip): ").strip()
        if data_path and os.path.exists(data_path):
            results_df = ablation.batch_analysis(data_path, num_samples=10)
    else:
        print("Files not found. Please check the paths.")
        print("\nTo use this script, you need:")
        print("1. A saved model checkpoint (e.g., best_fold0.pt)")
        print("2. The config JSON file used for training")
        print("3. A CSV file with SMILES column for testing")
