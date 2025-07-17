''' 
PharmHGT: A Hierarchical Graph Transformer for Drug-Target Interaction Prediction
This code is part of the PharmHGT project, which implements a hierarchical graph transformer model for Surfactants
Original code was adapted and modified to work with Surfactants data.
This file contains the training loop and evaluation functions for the PharmHGT model.

@adapted by: Gabi107
@date: 2024-07-19

'''


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

from data import create_dataloader
from model import PharmHGT as Model
from schedular import NoamLR
from utils import get_func,remove_nan_label, rmse

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


def train(data_args,train_args,model_args,seeds=[0,100,200,300,400]):
    
    epochs = train_args['epochs']
    device = train_args['device'] if torch.cuda.is_available() else 'cpu'
    save_path = train_args['save_path']

    # Early stopping parameters
    patience = train_args.get('patience', 50)  # Default patience of 50 epochs
    min_delta = train_args.get('min_delta', 1e-4)

    wandb.config = train_args

    os.makedirs(save_path,exist_ok=True)
    
    
    results = []
    for seed in seeds:
        # torch.manual_seed(seed)
        # Set seed for this fold
        set_seed(seed)

        for fold in range(train_args['num_fold']):
            wandb.init(project='PharmHGT', entity='entity_name',group=train_args["data_name"],name=f'seed{seed}_fold{fold}',reinit=True)
            trainloader = create_dataloader(data_args,f'{seed}_fold_{fold}_train.csv',shuffle=True)
            valloader = create_dataloader(data_args,f'{seed}_fold_{fold}_valid.csv',shuffle=False,train=False)
            testloader = create_dataloader(data_args,f'{seed}_fold_{fold}_test.csv',shuffle=False,train=False)
            print(f'dataset size, train: {len(trainloader.dataset)}, \
                    val: {len(valloader.dataset)}, \
                    test: {len(testloader.dataset)}')
            model = Model(model_args).to(device)

            # # Add debugging here for first fold of first seed
            # if seed == seeds[0] and fold == 0:
            #     print("\nDebug Information:")
            #     verify_data_dimensions(trainloader, model, device)
            #     print("\nContinuing with training...\n")

            # Initialize model weights deterministically
            model.apply(lambda m: torch.nn.init.xavier_normal_(m.weight.data) 
                       if isinstance(m, torch.nn.Linear) else None)

            optimizer = Adam(model.parameters())

            # Use MSE loss for regression
            loss_fn = nn.MSELoss()
            # Use RMSE for evaluation
            metric_fn = rmse
            
            scheduler = NoamLR(
                optimizer=optimizer,
                warmup_epochs=[train_args['warmup']],
                total_epochs=[epochs],
                steps_per_epoch=len(trainloader.dataset) // data_args['batch_size'],
                init_lr=[train_args['init_lr']],
                max_lr=[train_args['max_lr']],
                final_lr=[train_args['final_lr']]
            )

            best_rmse = float('inf')
            best_epoch = 0
            epochs_without_improvement = 0
            
            for epoch in tqdm(range(epochs)):
                model.train()
                total_loss = 0
                for bg,labels in trainloader:
                    bg= bg.to(device)
                    labels = labels.float().to(device)
                    pred = model(bg)
                    loss = loss_fn(pred, labels)
                    total_loss += loss.item()*len(labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                total_loss = total_loss / len(trainloader.dataset)
                
                # Validation phase
                model.eval()
                val_rmse = evaluate(valloader, model, device, metric_fn, torch.float32, 'regression')
                if val_rmse < best_rmse- min_delta:
                    best_rmse = val_rmse
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    # torch.save(model.state_dict(), os.path.join(save_path, f'best_fold{fold}.pt'))
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_rmse': best_rmse,
                        'seed': seed
                    }, os.path.join(save_path, f'best_fold{fold}.pt'))
                else:
                    epochs_without_improvement += 1   

                wandb.log({
                    'train MSE loss': round(total_loss, 4),
                    'valid RMSE': round(val_rmse, 4),
                    'lr': round(math.log10(scheduler.lr[0]), 4),
                    'epoch': epoch  # Add this line
                })

                # Early stopping check
                if epochs_without_improvement >= patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs')
                    break
                
            # Test phase
            checkpoint = torch.load(os.path.join(save_path, f'best_fold{fold}.pt'))
            model = Model(model_args).to(device)
            # state_dict = torch.load(os.path.join(save_path,f'./best_fold{fold}.pt'))
            # model.load_state_dict(state_dict)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            with torch.no_grad():
                test_rmse = evaluate(testloader, model, device, metric_fn, torch.float32, 'regression')
            results.append(test_rmse)

            print(f'Seed {seed}, Fold {fold}:')
            print(f'Best epoch {best_epoch} for fold {fold}, val RMSE: {best_rmse}, test RMSE: {test_rmse}')
            wandb.finish()
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
    results = train(data_args,train_args,model_args,seed)
    print(f'average performance: {np.mean(results)}+/-{np.std(results)}')