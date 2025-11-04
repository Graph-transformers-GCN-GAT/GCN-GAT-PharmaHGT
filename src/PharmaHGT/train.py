''' 
PharmHGT: A Hierarchical Graph Transformer for Drug-Target Interaction Prediction
This code is part of the PharmHGT project, which implements a hierarchical graph transformer model for Surfactants
Original code was adapted and modified to work with Surfactants data.
This file contains the training loop and evaluation functions for the PharmHGT model.

@adapted by: Gabi107
@date: 2024-07-19

'''
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import os
import math
import numpy as np
import pandas as pd
import json
import operator
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam
import wandb
import random
import dgl
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from data import create_dataloader
from model import PharmHGT as Model
from schedular import NoamLR
from utils import get_func,remove_nan_label, rmse

import datetime, shutil, hashlib
import glob

class MetricsTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_rmses = []
        self.train_r2s = []
        self.val_r2s = []
        self.train_rmses = []
        
    def update(self, train_loss, val_loss, val_rmse, train_rmse, train_r2, val_r2):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_rmses.append(val_rmse)
        self.train_rmses.append(train_rmse)
        self.train_r2s.append(train_r2)
        self.val_r2s.append(val_r2)

def plot_metrics(tracker, model_name, save_path, fold=None):
    epochs = range(1, len(tracker.train_losses) + 1)
    plt.figure(figsize=(15, 5))
    
    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, tracker.train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, tracker.val_losses, 'ro-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.title(f'{model_name} - {"Fold " + str(fold) if fold is not None else ""} - Loss')
    
    # Plot RMSE
    plt.subplot(1, 3, 2)
    plt.plot(epochs, tracker.train_rmses, 'bo-', label='Training RMSE')
    plt.plot(epochs, tracker.val_rmses, 'ro-', label='Validation RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title(f'{model_name} - {"Fold " + str(fold) if fold is not None else ""} - RMSE')
    
    # Plot R2
    plt.subplot(1, 3, 3)
    plt.plot(epochs, tracker.train_r2s, 'bo-', label='Training R²')
    plt.plot(epochs, tracker.val_r2s, 'ro-', label='Validation R²')
    plt.xlabel('Epochs')
    plt.ylabel('R² Score')
    plt.legend()
    plt.title(f'{model_name} - {"Fold " + str(fold) if fold is not None else ""} - R²')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'metrics_{"fold_" + str(fold) if fold is not None else "performance"}.png'))
    plt.close()

def set_seed(seed):
    """Set all seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    dgl.random.seed(seed)

    torch.cuda.manual_seed(seed)
    # Enforce deterministic algorithms
    torch.use_deterministic_algorithms(True)

def verify_data_dimensions(dataloader, model, device):
    for bg, labels in dataloader:
        bg = bg.to(device)  # Move graph to GPU
        labels = labels.to(device)  # Move labels to GPU
        print("Graph batch size:", bg.batch_size)
        print("Number of nodes:", bg.num_nodes())
        print("Node feature dimensions:", bg.nodes['a'].data['f'].shape)
        print("Label shape:", labels.shape)
        print("Label values:", labels[:5])  # Print first 5 labels
        
        # Test a forward pass
        out = model(bg)
        print("Model output shape:", out.shape)
        print("Model output values:", out[:5])
        break

def evaluate(dataloader, model, device, metric_fn, metric_dtype, task):
    metric = 0
    for bg, labels in dataloader:
        # Move both graph and labels to device
        bg = bg.to(device)
        labels = labels.type(metric_dtype).to(device)  # Move labels to device after type casting
        
        # Get predictions and move to CPU for evaluation
        pred = model(bg).cpu().detach()
        labels = labels.cpu()  # Move labels back to CPU for consistent evaluation
        
        if task == 'classification':
            pred = torch.sigmoid(pred)
        elif task == 'multiclass':
            pred = torch.softmax(pred, dim=1)
            
        num_task = pred.size(1)
        if num_task > 1:
            m = 0
            for i in range(num_task):
                try:
                    m += metric_fn(*remove_nan_label(pred[:,i], labels[:,i]))
                except:
                    print(f'only one class for task {i}')
            m = m/num_task
        else:
            m = metric_fn(pred, labels.reshape(pred.shape))
            
        metric += m.item() * len(labels)
    
    metric = metric/len(dataloader.dataset)
    return metric

def save_config_json(save_dir, original_config_path, resolved_config, seed, fold=None):
    os.makedirs(save_dir, exist_ok=True)
    # 1) the resolved config you actually used (includes seed, defaults, etc.)
    name = f"config_used_seed{seed}" + (f"_fold{fold}" if fold is not None else "") + ".json"
    with open(os.path.join(save_dir, name), "w") as f:
        json.dump(resolved_config, f, indent=2)

    # 2) optionally keep an exact copy of the original file for traceability
    try:
        shutil.copy2(original_config_path, os.path.join(save_dir, "config_original.json"))
    except Exception:
        pass

def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def snapshot_splits(data_args, seed, fold, out_dir):
    """
    Copy the exact CSVs used by the dataloaders and write a small manifest
    with counts and SHA256 hashes for reproducibility.
    """
    os.makedirs(out_dir, exist_ok=True)
    base = data_args['path']  # where your split CSVs live

    files = {
        "train": f"{seed}_fold_{fold}_train.csv",
        "valid": f"{seed}_fold_{fold}_valid.csv",
        "test":  f"{seed}_fold_{fold}_test.csv",
    }

    manifest = {"seed": seed, "fold": fold, "files": {}}

    for split, fname in files.items():
        src = os.path.join(base, fname)
        dst = os.path.join(out_dir, fname)  # keep the same name in your run folder

        # copy the original split file
        shutil.copy2(src, dst)

        # compute a quick summary for traceability
        df = pd.read_csv(dst)
        manifest["files"][split] = {
            "filename": fname,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "sha256": sha256sum(dst),
        }

    # write manifest.json
    with open(os.path.join(out_dir, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def train(data_args,train_args,model_args,seeds=[0,100,200,300,400], original_config_path=None):
    
    epochs = train_args['epochs']
    device = train_args['device'] if torch.cuda.is_available() else 'cpu'
    save_path = train_args['save_path']

    os.makedirs(save_path,exist_ok=True)

    results = []
    for seed in seeds:
        # make a per-seed subfolder (prevents overwrites)
        seed_dir = os.path.join(save_path, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        # save the resolved config used for THIS seed
        resolved = {
            "seed": seed,
            "data": data_args,
            "train": {k:v for k,v in train_args.items()},
            "model": model_args
        }
        save_config_json(seed_dir, original_config_path, resolved, seed)

        for fold in range(train_args['num_fold']):
            fold_dir = os.path.join(seed_dir, f"fold_{fold}")
            os.makedirs(fold_dir, exist_ok=True)

            # (optional) also save per-fold config snapshot
            save_config_json(fold_dir, original_config_path, resolved, seed, fold)

    # Early stopping parameters
    patience = train_args.get('patience', 50)  # Default patience of 50 epochs
    min_delta = train_args.get('min_delta', 1e-4)

    wandb.config = train_args

       
    
    # results = []
    for seed in seeds:
        # torch.manual_seed(seed)
        # Set seed for this fold
        set_seed(seed)

        for fold in range(train_args['num_fold']):
            metrics_tracker = MetricsTracker()

            wandb.init(project='PharmHGT', entity='entity_name',group=train_args["data_name"],name=f'seed{seed}_fold{fold}',reinit=True)
            trainloader = create_dataloader(data_args,f'{seed}_fold_{fold}_train.csv',shuffle=True)
            valloader = create_dataloader(data_args,f'{seed}_fold_{fold}_valid.csv',shuffle=False,train=False)
            testloader = create_dataloader(data_args,f'{seed}_fold_{fold}_test.csv',shuffle=False,train=False)
            
            # choose an output dir for this fold (recommended)
            fold_dir = os.path.join(train_args['save_path'], f"seed_{seed}", f"fold_{fold}")
            os.makedirs(fold_dir, exist_ok=True)

            # snapshot the exact CSVs used by these loaders
            snapshot_splits(data_args, seed, fold, fold_dir)

            print(f'dataset size, train: {len(trainloader.dataset)}, \
                    val: {len(valloader.dataset)}, \
                    test: {len(testloader.dataset)}')
            set_seed(seed)  # Re-seed before model creation
            model = Model(model_args).to(device)

            # # Add debugging here for first fold of first seed
            # if seed == seeds[0] and fold == 0:
            #     print("\nDebug Information:")
            #     verify_data_dimensions(trainloader, model, device)
            #     print("\nContinuing with training...\n")

            # Initialize model weights deterministically
            model.apply(lambda m: torch.nn.init.xavier_normal_(m.weight.data) 
                       if isinstance(m, torch.nn.Linear) else None)

            optimizer = Adam(model.parameters(), weight_decay=1e-4)

            # Use MSE loss for regression
            mse_loss = nn.MSELoss()

            # Use Huber loss (SmoothL1) for robust regression
            beta = train_args.get('huber_beta', 1.0)  # delta parameter; 1.0 is a good default
            huber = nn.SmoothL1Loss(beta=beta)
            loss_warmup_epochs = train_args.get('loss_warmup_epochs', 15)

            # Use RMSE for evaluation
            metric_fn = rmse
            
            scheduler = NoamLR(
                optimizer=optimizer,
                warmup_epochs=[train_args['warmup']],
                total_epochs=[epochs],
                # steps_per_epoch=len(trainloader.dataset) // data_args['batch_size'],
                steps_per_epoch=max(1, len(trainloader)),
                init_lr=[train_args['init_lr']],
                max_lr=[train_args['max_lr']],
                final_lr=[train_args['final_lr']]
            )

            ## ----- Snapshot-ensemble settings (top-K) -----
            top_k = train_args.get('top_k_snapshots', 1)  # # keep the best 5 by val RMSE
            best_list = []  # list of (val_rmse, epoch, state_dict)

            best_rmse = float('inf')
            best_epoch = 0
            epochs_without_improvement = 0
            
            for epoch in tqdm(range(epochs)):
                model.train()
                total_loss = 0
                train_preds = []
                train_labels = []
                for bg,labels in trainloader:
                    bg= bg.to(device)
                    labels = labels.float().to(device)
                    pred = model(bg)
                    # loss = loss_fn(pred, labels)
                    loss_fn_now = mse_loss if epoch < loss_warmup_epochs else huber
                    loss = loss_fn_now(pred, labels)

                    #Info for the plot
                    train_preds.extend(pred.detach().cpu().numpy())
                    train_labels.extend(labels.detach().cpu().numpy())

                    total_loss += loss.item()*len(labels)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                total_loss = total_loss / len(trainloader.dataset)
                
                #Info for the plot
                train_preds = np.array(train_preds)
                train_labels = np.array(train_labels)
                train_rmse = rmse(torch.tensor(train_preds), torch.tensor(train_labels))
                # train_r2 = r2_score(train_labels, train_preds)
                train_r2 = r2_score(train_labels.ravel(), train_preds.ravel())
                
                # Validation phase
                model.eval()
                val_preds = []
                val_labels = []
                val_loss = 0

                with torch.no_grad():
                    val_loss_fn = mse_loss if epoch < loss_warmup_epochs else huber
                    val_rmse = evaluate(valloader, model, device, metric_fn, torch.float32, 'regression')

                    # Collect predictions and labels
                
                    for bg, labels in valloader:
                        bg = bg.to(device)
                        labels = labels.float().to(device)
                        pred = model(bg)
                        loss = val_loss_fn(pred, labels)
                        val_loss += loss.item() * len(labels)

                        # Store predictions and labels for R2 calculation
                        val_preds.extend(pred.cpu().detach().numpy())
                        val_labels.extend(labels.cpu().detach().numpy())

                val_loss = val_loss / len(valloader.dataset)
                # val_r2 = r2_score(val_labels, val_preds)
                val_r2 = r2_score(np.asarray(val_labels).ravel(), np.asarray(val_preds).ravel())
                
                # Update metrics tracker
                metrics_tracker.update(total_loss, val_loss, val_rmse, train_rmse, train_r2, val_r2)
                
                # Plot current metrics
                if (epoch + 1) % 5 == 0:  # Plot every 5 epochs
                    plot_metrics(metrics_tracker, 'SurfactantModel', fold_dir, fold)


                if val_rmse < best_rmse- min_delta:
                    best_rmse = val_rmse
                    best_epoch = epoch
                    epochs_without_improvement = 0

                    ckpt_path = os.path.join(
                        fold_dir, f'best_fold{fold}.pt'
                    )
                    # torch.save(model.state_dict(), os.path.join(save_path, f'best_fold{fold}.pt'))
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        # 'optimizer_state_dict': optimizer.state_dict(),
                        # 'scheduler_state_dict': scheduler.state_dict(),
                        'best_rmse': best_rmse,
                        'seed': seed
                    # }, os.path.join(save_path, f'best_fold{fold}.pt'))
                    }, ckpt_path)

                    # keep only the best top_k snapshots
                    # best_list.append((val_rmse, ckpt_path))
                    # best_list = sorted(best_list, key=lambda x: x[0])[:top_k]

                    ckpt_min  = os.path.join(fold_dir, f'best_fold{fold}_minimal.pt')

                    torch.save(model.state_dict(), ckpt_min)
                else:
                    epochs_without_improvement += 1   

                wandb.log({
                    'train loss (Huber)': round(total_loss, 4),
                    'train RMSE': round(train_rmse.item(), 4),
                    'valid RMSE': round(val_rmse, 4),
                    'train R2': round(train_r2, 4),
                    'valid R2': round(val_r2, 4),
                    'lr': round(math.log10(scheduler.lr[0]), 4),
                    'epoch': epoch  # Add this line
                })

                # Early stopping check
                if epochs_without_improvement >= patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs')
                    break

            # Final metrics plot
            plot_metrics(metrics_tracker, 'SurfactantModel', fold_dir, fold)

            # Test phase
            # checkpoint = torch.load(os.path.join(save_path, f'best_fold{fold}.pt'))
            # model = Model(model_args).to(device)
            # # state_dict = torch.load(os.path.join(save_path,f'./best_fold{fold}.pt'))
            # # model.load_state_dict(state_dict)
            # model.load_state_dict(checkpoint['model_state_dict'])
            # model.eval()
            # with torch.no_grad():
            #     test_rmse = evaluate(testloader, model, device, metric_fn, torch.float32, 'regression')
            # results.append(test_rmse)

            # print(f'Seed {seed}, Fold {fold}:')
            # print(f'Best epoch {best_epoch} for fold {fold}, val RMSE: {best_rmse}, test RMSE: {test_rmse}')
            # wandb.finish()

            # ===== Test phase: load the single best checkpoint you saved =====
            ckpt_path = os.path.join(fold_dir, f'best_fold{fold}.pt')
            assert os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"

            best_model = Model(model_args).to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            best_model.load_state_dict(ckpt['model_state_dict'])
            best_model.eval()

            preds, labels_list = [], []
            with torch.no_grad():
                for bg, labels in testloader:
                    bg = bg.to(device)
                    labels = labels.float().to(device)
                    out = best_model(bg)
                    preds.append(out.detach().cpu())
                    labels_list.append(labels.detach().cpu())

            preds = torch.cat(preds, dim=0)
            labels_all = torch.cat(labels_list, dim=0)

            test_rmse = rmse(preds, labels_all).item()
            test_mae  = nn.L1Loss()(preds, labels_all).item()
            test_r2   = r2_score(labels_all.numpy(), preds.numpy())

            results.append(test_rmse)

            print(f'Seed {seed}, Fold {fold}:')
            print(f'Best (single) val RMSE @epoch {best_epoch}: {best_rmse:.4f}')
            print(f'Test (single best) → RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | R²: {test_r2:.3f}')

            wandb.log({
                'test RMSE (single)': round(test_rmse, 4),
                'test MAE (single)': round(test_mae, 4),
                'test R2 (single)': round(test_r2, 4),
                'best_epoch': best_epoch,
                'best_val_RMSE': round(best_rmse, 4)
            })
            wandb.finish()


            # # ===========================
            # # Test phase with snapshot ensemble (top-K)
            # # ===========================
            # # If nothing got saved into best_list (e.g., early exit), fall back to whats in fold_dir
            # if len(best_list) == 0:
            #     candidates = glob.glob(os.path.join(fold_dir, f'best_fold{fold}_epoch*_rmse*.pt'))
            #     assert len(candidates) > 0, f"No checkpoints found in {fold_dir}"
            #     # pick the best by parsing rmse out of filename
            #     def parse_rmse(p):
            #         try:
            #             return float(p.split('_rmse')[-1].replace('.pt', ''))
            #         except:
            #             return 1e9
            #     candidates = sorted(candidates, key=parse_rmse)
            #     best_list = [(parse_rmse(candidates[0]), candidates[0])]

            # # Accumulate predictions from each snapshot
            # snapshot_preds = []
            # with torch.no_grad():
            #     for _, ckpt_path in best_list:
            #         snap_model = Model(model_args).to(device)
            #         ckpt = torch.load(ckpt_path, map_location=device)
            #         snap_model.load_state_dict(ckpt['model_state_dict'])
            #         snap_model.eval()

            #         preds_this = []
            #         labels_this = []
            #         for bg, labels in testloader:
            #             bg = bg.to(device)
            #             labels = labels.float().to(device)
            #             pred = snap_model(bg)
            #             preds_this.append(pred.detach().cpu())
            #             labels_this.append(labels.detach().cpu())
            #         snapshot_preds.append(torch.cat(preds_this, dim=0))

            # # average predictions across snapshots
            # mean_pred = torch.stack(snapshot_preds, dim=0).mean(dim=0)
            # # labels from the last loop are fine (test set is fixed)
            # all_labels = torch.cat(labels_this, dim=0)

            # # Compute metrics
            # test_rmse = rmse(mean_pred, all_labels).item()
            # test_mae  = nn.L1Loss()(mean_pred, all_labels).item()
            # test_r2   = r2_score(all_labels.numpy(), mean_pred.numpy())

            # results.append(test_rmse)

            # print(f'Seed {seed}, Fold {fold}:')
            # print(f'Best (single) val RMSE @epoch {best_epoch}: {best_rmse:.4f}')
            # print(f'Ensemble ({len(best_list)} ckpts) Test → RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | R²: {test_r2:.3f}')

            # wandb.log({
            #     'test RMSE (ensemble)': round(test_rmse, 4),
            #     'test MAE (ensemble)': round(test_mae, 4),
            #     'test R2 (ensemble)': round(test_r2, 4),
            #     'snapshots_used': len(best_list)
            # })
            # wandb.finish()

    return results


if __name__=='__main__':


    import sys
    config_path = sys.argv[1]
    config = json.load(open(config_path,'r'))
    
    seed = config['seed']
    if not isinstance(seed,list):
        seed = [seed]

    # Set the seed for the entire system
    set_seed(seed[0])  # In your case, this is 2022
    
    data_args = config['data']
    train_args = config['train']
    train_args['data_name'] = config_path.split('/')[-1].strip('.json')
    model_args = config['model']
    
    
    print(config)
    results = train(data_args,train_args,model_args,seed, original_config_path=config_path)
    print(f'average performance: {np.mean(results)}+/-{np.std(results)}')