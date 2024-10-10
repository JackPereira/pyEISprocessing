# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn
from sklearn import metrics

'''
======================================
Basic functions to add pyTorch functionality
======================================
'''
def get_device():
    '''
    Gets device to run pyTorch training on. Defaults to CUDA if available.
    See: https://pytorch.org/get-started/locally/ and https://developer.nvidia.com/cuda-zone
    for overview of CUDA-capable systems.
    '''
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return device

def ctd(tensor):
    return tensor.device

def tensor_transform(np_array, dtypes=torch.float):
    '''
    Transforms numpy.ndarray into a tensor with proper dtype.

    Parameters
    ----------
    np_array : numpy.ndarray
        Input array.
    dtypes : torch.cfloat, torch.cdouble, torch.float, torch.dobule, optional
        Output tensor datatype.

    Returns
    -------
    tens : dtypes
        Transformed tensor.
    '''
    tens = torch.from_numpy(np_array)
    tens.to(dtype=dtypes)
    return tens

def train_loop(dataloader, model, loss_fn, optimizer,loss_weights, dtypes = torch.float,batch_size=100, verbose=True,device=get_device()):
    '''
    Basic training structure for pyTorch models using dataloader batches.

    Parameters
    ----------
    dataloader : torch.utils.data.Dataloader()
        DataLoader class containing training data.
    model : torch.nn.Module
        The pyTorch model being trained.
    loss_fn : torch.nn.Module._Loss
        Loss function used to optimize model.
    optimizer : torch.nn.optim.Optimizer
        Optimizer used to tune model parameters during training.
    loss_weights : np.ndarray
        Weighting scheme used to normalize targets in the loss function.
        Important for loss functions such as L1 that do not account for discrepancies
        between target scales during calculation.
        Used by default in pyt_train by taking the mean of each target. Can be passed
        directly using the kwarg 'loss_weights'.
    dtypes : dtype, optional
        dtype for all tensors and model parameters.
    batch_size : int, optional
        Size of training batches grabbed from the dataloader.
    verbose : bool, optional
        If True, add print statements during training.
    device : 
        Device to perform training operations in. See get_device().
    '''
    size = len(dataloader.dataset)
    if not loss_weights == 'batch mean':
        batch_mean = False
    else:
        batch_mean = True
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(dtype=dtypes)
        pred = model(X.to(device))
        if batch_mean:
            loss_weights = tensor_transform(np.mean(y.numpy(force=True),axis=0), dtypes=dtypes)
        loss = 0
        loss += loss_fn(pred/loss_weights.to(device),y.to(device)/loss_weights.to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn, loss_weights,dtypes=torch.float,verbose=True,device=get_device()):
    model.eval()
    num_batches = len(dataloader)
    if not loss_weights == 'batch mean':
        loss_weights = loss_weights.to(device)
        batch_mean = False
    else:
        batch_mean = True
    test_loss = 0
    #for param in model.parameters():
        #param.data = param.data.to(torch.cfloat)
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device,dtype=dtypes))
            if batch_mean:
                loss_weights = tensor_transform(np.mean(y.numpy(force=True),axis=0), dtypes=dtypes)
            test_loss += loss_fn(pred/loss_weights.to(device), y.to(device,dtype=dtypes)/loss_weights.to(device)).item()
            if verbose:
                pred_np = pred.detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()
                #if not model.is_complex_:
                r2 = metrics.r2_score(y_np,pred_np)
                #else:
                    #r2 = metrics.r2_score(np.real(y_np),np.real(pred_np))
    if verbose:
        test_loss /= num_batches
        print(f"Test Error: \n Accuracy: {(r2):>0.3f}, Avg loss: {test_loss:>8f} \n")
    return test_loss

class Data(Dataset):
    def __init__(self, feature_set, target_set,dtypes):
        self.features_ = tensor_transform(feature_set,dtypes)
        self.targets_ = tensor_transform(target_set,dtypes)
    
    def __len__(self):
        return np.size(self.features_,axis=0)
    
    def __getitem__(self, idx):
        feature = self.features_[idx,:]
        target = self.targets_[idx,:]
        return feature, target
    
'''
======================================
Prebuilt nn.Module models that I've found perform well 
for relatively simple equivalent circuits.
Additionally, some beta functionality for complex inputs are 
under development here.
======================================
'''

class ComplexReLU(nn.Module):
    def forward(self, input):
        return input * (input.real > 0).float()

class ComplexDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        if self.training:
            mask = (torch.rand(input.size()) > self.p).to(input.device)
            return input * mask / (1 - self.p)  # Scale to keep expected value
        return input
    
class MultioutputRegression(nn.Module):
    def __init__(self, in_features, middle_neurons, out_features, p):
        super().__init__()
        self.linrelu = nn.Sequential(
            nn.Linear(in_features,out_features=middle_neurons),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(middle_neurons, in_features),
            nn.ReLU(),
            nn.Linear(in_features, out_features),   
        )
    
    def forward(self, x):
        logits = self.linrelu(x)
        return logits

class MOR_3L(nn.Module):
    def __init__(self, in_features, mn1,mn2, out_features, p):
        super().__init__()
        self.linrelu = nn.Sequential(
            nn.Linear(in_features,out_features=mn1),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(mn1, mn2),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(in_features=mn2,out_features=in_features),
            nn.ReLU(),
            nn.Linear(in_features, out_features),   
        )
    
    def forward(self, x):
        logits = self.linrelu(x)
        return logits