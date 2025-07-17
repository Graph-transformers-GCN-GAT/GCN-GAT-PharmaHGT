# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:50:47 2020
Modified on Fri Jul 19 10:00:00 2024

@author: sqin34
@modified by: Gabi107
"""

from __future__ import absolute_import

import pickle, time
import torch

import os

import argparse, os, time, random, pickle, csv
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch.optim
import torch.utils.data

from dgllife.utils import EarlyStopping
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score

class AccumulationMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0.0
        self.avg = 0.0
        self.sum = 0
        self.count = 0.0
        self.predictions = [] #Added
        self.true_labels = [] #Added

    #def update(self, value, n=1):
    def update(self, value, n=1, preds=None, labels=None): #Modified
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
        self.sqrt = self.value ** 0.5
        self.rmse = self.avg ** 0.5
        if preds is not None and labels is not None:   # Added
            self.predictions.extend(preds.cpu().detach().numpy())
            self.true_labels.extend(labels.cpu().numpy())



def print_result(stage, epoch, i, data_loader, batch_time, loss_accum):
    print('Epoch: [{0}][{1}/{2}]\t'
          'Time {batch_time.value:.3f} ({batch_time.avg:.3f})\t'
          'Loss {loss.value:.2f} ({loss.avg:.2f})\t'
          'RMSE {loss.sqrt:.2f} ({loss.rmse:.2f})\t'.format(
        epoch + 1, i, len(data_loader), batch_time=batch_time,
        loss=loss_accum))


def print_final_result(stage, epoch, lr, loss_accum):
    if stage == 'train':
        print("[Stage {:s}]: Epoch {:d} finished with lr={:f} loss={:.3f}".format(
            stage, epoch + 1, lr, loss_accum.avg))
    if stage == 'validate':
        print('******RMSE {loss.rmse:.3f}'.format(loss=loss_accum))


def write_result(writer, stage, loss_accum, epoch, num_iters, i, lr, fname):
    if stage == 'train':
        writer.add_scalar('training_loss_{}'.format(fname),
                          loss_accum.value, epoch * num_iters + i)
        writer.add_scalar('learning_rate_{}'.format(fname),
                          lr, epoch * num_iters + i)


def write_final_result(writer, stage, loss_accum, epoch, fname):
    if stage == 'train':
        writer.add_scalars('rmse_RMSE_{}'.format(fname),
                           {"train": loss_accum.rmse}, epoch + 1)
    if stage == 'validate':
        writer.add_scalars('rmse_RMSE_{}'.format(fname), {"val": loss_accum.rmse}, epoch + 1)

    if stage == 'testtest':
        writer.add_scalars('rmse_RMSE_{}'.format(fname), {"testtest": loss_accum.rmse}, epoch + 1)
    if stage == 'testtrain':
        writer.add_scalars('rmse_RMSE_{}'.format(fname), {"testtrain": loss_accum.rmse}, epoch + 1)
    if stage == 'testval':
        writer.add_scalars('rmse_RMSE_{}'.format(fname), {"testval": loss_accum.rmse}, epoch + 1)

def save_prediction_result(args, y_pred, y_true, fname, stage, model_path):
    if stage == 'testtest':
        with open('{}/prediction_test_{}.pickle'.format(model_path, fname), 'wb') as f:
            pickle.dump([y_pred, y_true], f)
    if stage == 'testtrain':
        with open('{}/prediction_train_{}.pickle'.format(model_path, fname), 'wb') as f:
            pickle.dump([y_pred, y_true], f)
    if stage == 'testval':
        with open('{}/prediction_val_{}.pickle'.format(model_path, fname), 'wb') as f:
            pickle.dump([y_pred, y_true], f)

def save_saliency_result(args, saliency_map, fname, stage, model_path):
    if stage == 'testtest':
        with open('{}/saliency_test_{}.pickle'.format(model_path, fname), 'wb') as f:
            pickle.dump(saliency_map, f)
    if stage == 'testtrain':
        with open('{}/saliency_train_{}.pickle'.format(model_path, fname), 'wb') as f:
            pickle.dump(saliency_map, f)
    if stage == 'testval':
        with open('{}/saliency_val_{}.pickle'.format(model_path, fname), 'wb') as f:
            pickle.dump(saliency_map, f)


# Save check point
def save_checkpoint(state, fname, model_path):
    # skip the optimization state
    state.pop('optimizer', None)
    # torch.save(state, r'{}/{}.pth.tar'.format(model_path, fname))
    #for training only
    save_path = os.path.join(model_path, f'{fname}.pth.tar')
    print(f"Saving checkpoint to: {save_path}")
    torch.save(state, save_path)
# Train function
def train(train_loader, model, loss_fn, optimizer, epoch, args, fname, writer):
    """Train the network on the training set."""
    stage = "train"
    num_iters = len(train_loader)
    lr = args.lr
    batch_time = AccumulationMeter()
    loss_accum = AccumulationMeter()
    model.train()
    end = time.time()

    for i, data in enumerate(train_loader):
        if args.binary_system:
            if args.add_features:
                if args.rdkit_descriptor:
                    (graph1, graph2, descriptor1, descriptor2, label) = data
                    graph1 = graph1.to(args.gpu)
                    graph2 = graph2.to(args.gpu)
                    output = model((graph1, graph2), (descriptor1, descriptor2))
                else:
                    #print('Training with binary system and additional features not rdkit descriptor')
                    (graph1, graph2, descriptor, label) = data
                    graph1 = graph1.to(args.gpu)
                    graph2 = graph2.to(args.gpu)
                    output = model((graph1, graph2), descriptor)
            
            else:    
                (graph1, graph2, label) = data
                graph1 = graph1.to(args.gpu)
                graph2 = graph2.to(args.gpu)     
                output = model((graph1, graph2))
        else:
            if args.add_features:
                graph, descriptor, label = data
                graph = graph.to(args.gpu)
                descriptor = descriptor.to(args.gpu)
                output = model(graph, descriptor)
            else:
                graph, label = data
                graph = graph.to(args.gpu)
                output = model(graph)

        if args.gpu >= 0:
            label = label.cuda(args.gpu, non_blocking=True)

        loss = loss_fn(output, label.float())
        #loss_accum.update(loss.item(), label.size(0))
        loss_accum.update(loss.item(), label.size(0), output, label) #modified
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_result(stage, epoch, i, train_loader, batch_time, loss_accum)
            write_result(writer, stage, loss_accum, epoch, num_iters, i, lr, fname)

    # Calculate R2 score at the end of the epoch
    r2 = r2_score(loss_accum.true_labels, loss_accum.predictions)  #Added

    print_final_result(stage, epoch, lr, loss_accum)
    write_final_result(writer, stage, loss_accum, epoch, fname)
    return loss_accum.avg, loss_accum.rmse, r2     #loss_accum.rmse, r2 added


# Validate function
def validate(val_loader, model, epoch, args, fname, writer):
    """Evaluate the network on the entire validation set."""
    stage = 'validate'
    lr = args.lr
    batch_time = AccumulationMeter()
    loss_accum = AccumulationMeter()
    model.eval()
    with torch.set_grad_enabled(False):
        end = time.time()

        for i, data in enumerate(val_loader):
            if args.binary_system:
                if args.add_features:
                    if args.rdkit_descriptor:
                        (graph1, graph2, descriptor1, descriptor2, label) = data
                        graph1 = graph1.to(args.gpu)
                        graph2 = graph2.to(args.gpu) 
                        output = model((graph1, graph2), (descriptor1, descriptor2))
                    else:
                        #print('Validating with binary system and additional features not rdkit descriptor')
                        (graph1, graph2, descriptor, label) = data
                        graph1 = graph1.to(args.gpu)
                        graph2 = graph2.to(args.gpu) 
                        output = model((graph1, graph2), descriptor)
                else:    
                    (graph1, graph2, label) = data
                    graph1 = graph1.to(args.gpu)
                    graph2 = graph2.to(args.gpu) 
                    output = model((graph1, graph2))
            else:
                if args.add_features:
                    graph, descriptor, label = data
                    graph = graph.to(args.gpu)
                    output = model(graph, descriptor)
                else:
                    graph, label = data
                    graph = graph.to(args.gpu)
                    output = model(graph)

            if args.gpu >= 0:
                label = label.cuda(args.gpu, non_blocking=False)
          
            loss_fn = nn.MSELoss()
            loss = loss_fn(output, label.float())
            #loss_accum.update(loss.item(), label.size(0))
            loss_accum.update(loss.item(), label.size(0), output, label) #Added
            #torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print_result(stage, epoch, i, val_loader, batch_time, loss_accum)

    # Calculate R2 score at the end of the epoch
    r2 = r2_score(loss_accum.true_labels, loss_accum.predictions) #added

    print_final_result(stage, epoch, lr, loss_accum)
    write_final_result(writer, stage, loss_accum, epoch, fname)
    return loss_accum.rmse, loss_accum.avg, r2 #loss_accum.avg, r2 added


# Test function
def predict(test_loader, model, epoch, args, fname, stage, writer, model_path):
    """Evaluate the network on the entire test set."""
    lr = args.lr
    batch_time = AccumulationMeter()
    loss_accum = AccumulationMeter()
    model.eval()

    with torch.set_grad_enabled(True):
        end = time.time()
        y_pred = []
        y_true = []
        saliency_map = []

        for i, data in enumerate(test_loader):
            
            if args.binary_system:
                if args.add_features:
                    if args.rdkit_descriptor:
                        (graph1, graph2, descriptor1, descriptor2, label) = data
                        output, grad1, grad2 = model((graph1, graph2), (descriptor1, descriptor2))
                    else:
                        #print('Predicting with binary system and additional features not rdkit descriptor')
                        (graph1, graph2, descriptor, label) = data
                        graph1 = graph1.to(args.gpu)
                        graph2 = graph2.to(args.gpu)
                        #output = model((graph1, graph2), descriptor)
                        output, grad1, grad2 = model((graph1, graph2), descriptor)
                else:    
                    (graph1, graph2, label) = data
                    graph1 = graph1.to(args.gpu)
                    graph2 = graph2.to(args.gpu)
                    output, grad1, grad2 = model((graph1, graph2))
            else:
                if args.add_features:
                    graph, descriptor, label = data
                    graph = graph.to(args.gpu)
                    output, grad = model(graph, descriptor)
                else:
                    graph, label = data
                    graph = graph.to(args.gpu)
                    output, grad = model(graph)

            if args.gpu >= 0:
                label = label.cuda(args.gpu, non_blocking=False)

            label = label.view(-1,1)            
            y_true.append(label.float())
            y_pred.append(output)
            loss_fn = nn.MSELoss()
            loss = loss_fn(output, label.float())
            loss_accum.update(loss.item(), label.size(0))
            #torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

            # SALIENCY MAP
            if args.binary_system:
                if args.rdkit_descriptor:
                    saliency_map.append([grad1, grad2])
                else:
                    saliency_map.append(0)
            else:
                saliency_map.append(grad)

            if i % args.print_freq == 0:
                print_result(stage, epoch, i, test_loader, batch_time, loss_accum)

    print_final_result(stage, epoch, lr, loss_accum)
    write_final_result(writer, stage, loss_accum, epoch, fname)

    save_prediction_result(args, y_pred, y_true, fname, stage, model_path)
    save_saliency_result(args, saliency_map, fname, stage, model_path)
    return

# convert graph to data loader
def to_loader(dataset, collate, batch_size=1, shuffle=False):

    return torch.utils.data.DataLoader(dataset, batch_size = batch_size,
                                                       collate_fn = collate,
                                                       shuffle = shuffle)