'''
Jack Pereira
Department of Chemical Engineering
University of Washington
2024
'''
import warnings
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant 

import dataset_processing as dtsp
import pytorch_integration as pyti
import plotgen
from analysis import complex_to_polar
from analysis import polar_to_complex

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn import metrics
from sklearn.base import is_classifier
from sklearn.base import is_regressor

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torch import optim

'''
======================================
pyTorch preprocessing and training
======================================
'''

def pyt_preprocess(ModelDS_obj, DS_obj, name=None, cast_as_complex = False, test_size=0.2,
                   scale_data=True,scaler=MaxAbsScaler(),verbose=True,**kwargs):
    if not ModelDS_obj.pytorch_:
        raise Exception('pyt_preprocess can only be used on pyTorch neural networks (subclasses of nn.Module)')
    data_select = kwargs.get('data_select','all')
    ModelDS_obj.load_data(DS_obj,data_select=data_select)
    if verbose:
        print('Data loaded\n-------------')
    random_state = kwargs.get('random_state',None)
    
    if cast_as_complex: 
    # Complex support should be treated as a beta feature, as it is in pyTorch.
    # While the functionality is implemented here, and it works with certain custom Modules I've written,
    # these implementations currently rely on discarding the imaginary component or its loss gradient.
    # (which sorta defeats the whole purpose of wrapping both Zre and Zim into a single value,
    # as it's essentially unnecessary data loss when compared to split Re/Im float representations)
    # See: https://pytorch.org/docs/stable/complex_numbers.html
        ModelDS_obj.is_complex_ = True
        ModelDS_obj.pyt_train_test_split(test_size,random_state=random_state)
        if verbose:
            print('Data split into train/test sets\nTest size = {:.1%}\n-------------'.format(test_size))
        if scale_data:
            ModelDS_obj.pyt_scale_data(scaler)
            if verbose:
                print('Data scaled\n-------------')
                
    else:
        ModelDS_obj.is_complex_ = False
        ModelDS_obj.split_re_im()
        ModelDS_obj.train_test_split(test_size,random_state=random_state)
        if verbose:
            print('Data split into train/test sets\nTest size = {:.1%}\n-------------'.format(test_size))
        if scale_data:
            ModelDS_obj.scale_data(scaler)
            if verbose:
                print('Data scaled\n-------------')
    
def pyt_train(ModelDS_obj,learning_rate = 1e-3, epochs=50, batch_size=100,
              dynamic_lr=False, dlr_epoch= 30, dlr_scheduler='exp',
              scoring=True, score_method = 'default', plotting=False,verbose=True,**kwargs):
    '''
    TO DO:
        1. Handle different optimizers:
            1a. ISSUE: I don't think it's possible to pass optim.Optimizer directly, 
            as it would be initialized before the model is sent to device.
            1b. https://discuss.pytorch.org/t/should-i-create-optimizer-after-sending-the-model-to-gpu/133418
            It may be possible? NeeDS_obj testing
        2.  Handle schedulers better:
            3a. Users should be able to pass any arbitrary scheduler

    Parameters
    ----------
    ModelDS_obj : TYPE
        DESCRIPTION.
    learning_rate : TYPE, optional
        DESCRIPTION. The default is 1e-3.
    epochs : TYPE, optional
        DESCRIPTION. The default is 50.
    batch_size : TYPE, optional
        DESCRIPTION. The default is 100.
    dynamic_lr : TYPE, optional
        DESCRIPTION. The default is False.
    scoring : TYPE, optional
        DESCRIPTION. The default is True.
    plotting : TYPE, optional
        DESCRIPTION. The default is False.
    **kwargs : TYPE
        DESCRIPTION.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    if not ModelDS_obj.pytorch_:
        raise Exception('pyt_train can only be used on pyTorch neural networks (subclasses of nn.Module)')
    if verbose:
        print('Training model\n-------------')
    if ModelDS_obj.is_complex_:
        dtypes = torch.cfloat
    else:
        dtypes = torch.float
    device = pyti.get_device()
    if verbose:
        print('Using {} device\n-------------'.format(device))
    model = ModelDS_obj.model_.to(device,dtype=dtypes)
    for param in model.parameters():
        param.data = param.data.to(dtypes)
        
    loss = kwargs.get('loss_fn',nn.L1Loss(reduction='mean'))
    if isinstance(loss,_Loss):
        loss.to(device)
    else:
        warnings.warn('Invalid loss_fn input; ensure the function is a subclass of nn.Module._Loss. Defaulting to L1Loss',UserWarning)
        loss = nn.L1Loss(reduction='mean').to(device)
        
    if 'loss_weights' in kwargs:
        target_weights = kwargs['loss_weights']
        if isinstance(target_weights,list):
            target_weights = np.array(target_weights)
        elif target_weights is None or target_weights == 'none':
            target_weights = np.ones(np.size(ModelDS_obj.p_train_,axis=1),1)
        elif isinstance(target_weights,str):
            if not target_weights == 'batch mean' or not target_weights == 'none':
                raise Exception('Available loss_weights string inputs: batch mean or none')
        else:
            raise Exception('Invalid loss_weights input')
    else:
        target_weights = pyti.tensor_transform(np.mean(ModelDS_obj.p_train_,axis=0),dtypes)
        
    training_data = pyti.Data(ModelDS_obj.Z_train_,ModelDS_obj.p_train_,dtypes=dtypes)
    testing_data = pyti.Data(ModelDS_obj.Z_test_,ModelDS_obj.p_test_,dtypes=dtypes)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)
    
    allscores = np.empty((epochs,1))
    allloss = np.empty((epochs,1))
    
    if not dynamic_lr:
        optimizer = optim.AdamW(model.parameters(),lr=learning_rate)
        for t in range(epochs):
            if verbose:
                print(f"Epoch {t+1}\n-------------------------------")
            pyti.train_loop(train_dataloader,model,loss,optimizer,target_weights,batch_size=batch_size,dtypes=dtypes,verbose=verbose)
            closs, cscore = pyti.test_loop(test_dataloader,model,loss,target_weights,dtypes=dtypes,verbose=verbose,return_score=True)
            allscores[t] = cscore
            allloss[t] = closs
    else:
        optimizer = optim.AdamW(model.parameters(),lr=learning_rate)
        if dlr_scheduler == 'exp':
            gamma = kwargs.get('gamma',0.95)
            if verbose:
                print('Dynamic learning rate - Exponential decay\nGamma = {:.3f}\n-------------------------------'.format(gamma))
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=gamma)
        if dlr_scheduler == 'plateau':
            patience = kwargs.get('patience',10)
            factor = kwargs.get('factor',0.1)
            threshold = kwargs.get('threshold',0.0001)
            threshold_mode = kwargs.get('threshold_mode','rel')
            cooldown = kwargs.get('cooldown',0)
            min_lr = kwargs.get('min_lr',0)
            eps = kwargs.get('eps',1e-8)
            if verbose:
                print('Dynamic learning rate - Reduce on plataeu')
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=factor,patience=patience,threshold=threshold,
                                                             threshold_mode=threshold_mode, cooldown=cooldown,min_lr=min_lr,eps=eps
            )
            old_lr = learning_rate
        for t in range(epochs):
            if verbose:
                print(f"Epoch {t+1}\n-------------------------------")
            pyti.train_loop(train_dataloader, model, loss, optimizer, target_weights, batch_size=batch_size,device=device)
            closs, cscore = pyti.test_loop(test_dataloader,model,loss,target_weights,dtypes=dtypes,verbose=verbose,return_score=True)
            if dlr_scheduler != 'plateau':
                scheduler.step()
            elif dlr_scheduler == 'plateau':
                scheduler.step(closs)
                current_lr = scheduler.get_last_lr()[0]
                if verbose:
                    if current_lr != old_lr:
                        print('New learning rate: {:.3e}'.format(current_lr))
                        old_lr = current_lr
            allscores[t] = cscore
            allloss[t] = closs
            
    p_test_pred = model(torch.from_numpy(ModelDS_obj.Z_test_).to(device,dtype=dtypes))
    p_test_pred = p_test_pred.detach().cpu().numpy()
    p_train_pred = model(torch.from_numpy(ModelDS_obj.Z_train_).to(device,dtype=dtypes))
    p_train_pred = p_train_pred.detach().cpu().numpy()
    
    ModelDS_obj.model_.load_state_dict(model.state_dict())
    ModelDS_obj.p_test_pred_ = p_test_pred
    ModelDS_obj.p_train_pred_ = p_train_pred
    del p_test_pred
    del p_train_pred
    ModelDS_obj.score_ = allscores
    ModelDS_obj.loss_ = allloss
    
    if scoring:
        get_scores(ModelDS_obj,score_method,verbose)
    if plotting:
        titles = kwargs.get('titles',ModelDS_obj.params_names_)
        axes_units = kwargs.get('axes_units',ModelDS_obj.params_units_)
        plotgen.plot_results(titles,axes_units,ModelDS_obj.p_test_,ModelDS_obj.p_test_pred_,ModelDS_obj.name_,ModelDS_obj.dataname_)

def pyt_preprocess_and_train(ModelDS_obj, DS_obj, name=None, cast_as_complex = False, test_size=0.2, scale_data=True, 
                             learning_rate = 1e-3, epochs=50, batch_size=100, dynamic_lr = False, scoring=True, 
                             scoring_method='default',plotting=False,savefig=True,savepath='',verbose=True,**kwargs):
    if not ModelDS_obj.pytorch_:
        raise Exception('pyt_preprocess_and_train can only be used on pyTorch neural networks (subclasses of nn.Module)')
    
    pyt_preprocess(ModelDS_obj, DS_obj, name, cast_as_complex,test_size,scale_data,verbose=verbose,**kwargs)
    pyt_train(ModelDS_obj,learning_rate=learning_rate,epochs=epochs,dynamic_lr=dynamic_lr,batch_size=batch_size,
              scoring=scoring,scoring_method=scoring_method,plotting=plotting,verbose=verbose,**kwargs)

'''
======================================
Scikit-learn preprocessing and training
======================================
'''

def preprocess(ModelDS_obj, DS_obj, name=None,test_size=0.2,scale_data=True,scaler=MaxAbsScaler(),
               verbose=True,**kwargs):
    if ModelDS_obj.pytorch_:
        raise Exception('For pyTorch modules, use pyt_preprocess')
    data_select = kwargs.get('data_select','all')
    ModelDS_obj.load_data(DS_obj,data_select=data_select)
    ModelDS_obj.split_re_im()
    if verbose:
        print('Data loaded and split into real and imaginary components\n-------------')
    random_state = kwargs.get('random_state',None)
    ModelDS_obj.train_test_split(test_size,random_state=random_state)
    if verbose:
        print('Data split into train/test sets\nTest size = {:.1%}\n-------------'.format(test_size))
    ModelDS_obj.scale_data(scaler)
    if verbose:
        print('Data scaled\n-------------')
       
def train(ModelDS_obj,score_method='default',plotting=False,verbose=True,**kwargs):
    if ModelDS_obj.pytorch_:
        raise Exception('For pyTorch modules, use pyt_preprocess_and_train')
            
    Z_train, Z_test, p_train = ModelDS_obj.Z_train_, ModelDS_obj.Z_test_, ModelDS_obj.p_train_
    ModelDS_obj.model_.fit(Z_train, p_train)
    p_train_pred = ModelDS_obj.model_.predict(Z_train)
    p_test_pred = ModelDS_obj.model_.predict(Z_test)
    ModelDS_obj.p_train_pred_ = p_train_pred
    ModelDS_obj.p_test_pred_ = p_test_pred
    
    if score_method is not None:
        get_scores(ModelDS_obj,score_method,verbose)
    if plotting:
        titles = kwargs.get('titles',ModelDS_obj.params_names_)
        axes_units = kwargs.get('axes_units',ModelDS_obj.params_units_)
        plotgen.plot_results(titles,axes_units,ModelDS_obj.p_test_,ModelDS_obj.p_test_pred_,ModelDS_obj.name_,ModelDS_obj.dataname_)

def preprocess_and_train(ModelDS_obj, DS_obj, name=None, test_size=0.2,scale_data=True, scaler=MaxAbsScaler(),
                         score_method = 'default', plotting=False,savefig=True,savepath='',verbose=True,**kwargs):
    if ModelDS_obj.pytorch_:
        raise Exception('For pyTorch modules, use pyt_preprocess_and_train')  
    preprocess(ModelDS_obj, DS_obj, name, test_size,scale_data,scaler,verbose=verbose,**kwargs)
    train(ModelDS_obj,score_method,plotting,verbose,**kwargs)
    if score_method is not None:
        get_scores(ModelDS_obj,score_method,verbose,**kwargs)
        
def get_scores(ModelDS_obj,score_method = 'default', verbose=True,**kwargs):
    # Classifier scoring
    if ModelDS_obj.model_type_ == 'classifier':
        if score_method == 'default':
            score_method = 'accuracy'
        elif score_method == 'accuracy':
            train_score = metrics.accuracy_score(ModelDS_obj.p_train_, ModelDS_obj.p_train_pred_)
            test_score = metrics.accuracy_score(ModelDS_obj.p_test_, ModelDS_obj.p_test_pred_)
        else:
            raise NotImplementedError('Other metrics will be implemented later')
    # Regressor scoring
    elif ModelDS_obj.model_type_ == 'regressor':
        if score_method == 'default':
            score_method = 'r2'
        if score_method == 'r2':
            train_score = metrics.r2_score(ModelDS_obj.p_train_, ModelDS_obj.p_train_pred_)
            test_score = metrics.r2_score(ModelDS_obj.p_test_, ModelDS_obj.p_test_pred_)
        elif score_method == 'rmse':
            train_score = metrics.root_mean_squared_error(ModelDS_obj.p_train_, ModelDS_obj.p_train_pred_)
            test_score = metrics.root_mean_squared_error(ModelDS_obj.p_test_, ModelDS_obj.p_test_pred_)
        else:
            print(score_method)
            raise NotImplementedError('Other metrics will be implemented later')
    ModelDS_obj.train_score_ = train_score
    ModelDS_obj.test_score_ = test_score

'''
======================================
Combined model-dataset class used as the basis of storing, 
training, and evaluating model-dataset pairs.
======================================
'''

class ModelDS():
    def __init__(self, model, pyt_mode = 'regression', name=None):
        if isinstance(model, nn.Module):
            self.pytorch_ = True
            if pyt_mode == 'regression':
                self.model_type_ = 'regressor'
            elif pyt_mode == 'classification':
                self.model_type_ = 'classifier'
            else:
                raise Exception('Invalid pyt_mode input. Valid inputs: regression; classification')
        elif is_classifier(model):
            self.model_type_ = 'classifier'
            self.pytorch_ = False
        elif is_regressor(model):
            self.model_type_ = 'regressor'
            self.pytorch_ = False
        else:
            raise NotImplementedError('Custom models not supported yet. Please pass an sklearn or pyTorch model')
        if name is None:
            self.name_ = self.model_type_
        else:
            self.name_ = name
        self.model_ = model
        self.contains_data_ = False
        self.is_split_ = False
        self.is_scaled_ = False
        self.is_preprocessed_ = False
        self.is_trained_ = False
    
    def swap_model(self,model,pyt_mode = 'regression',name=None):
        if isinstance(model, nn.Module):
            self.pytorch_ = True
            if pyt_mode == 'regression':
                self.model_type_ = 'regressor'
            elif pyt_mode == 'classification':
                self.model_type_ = 'classifier'
            else:
                raise Exception('Invalid pyt_mode input. Valid inputs: regression; classification')
        elif is_classifier(model):
            self.model_type_ = 'classifier'
            self.pytorch_ = False
        elif is_regressor(model):
            self.model_type_ = 'regressor'
            self.pytorch_ = False
        if name is None:
            self.name_ = self.model_type_
        else:
            self.name_ = name
        self.model_ = model
    
    def load_data(self,DS_obj, data_select = 'all'):
        if isinstance(DS_obj,dtsp.DS):
            if self.contains_data_:
                raise Exception('Data already found; run Model.clear_data before loading again')
            if isinstance(DS_obj,dtsp.EISdataset):
                Z = DS_obj.Z_var_
            else:
                Z = DS_obj.DRTdata_
                
            if data_select == 'noadded':
                if DS_obj.Z_var_ is None:
                    raise Exception('No Z data found in DS_obj')
                self.Z_ = Z
                self.tags_ = DS_obj.tags_
            elif data_select == 'added':
                if DS_obj.ap_ is None:
                    raise Exception('No added parameters found in DS_obj')
                self.Z_ = DS_obj.ap_
                self.tags_ = DS_obj.ap_tags_
            elif data_select == 'all':
                if DS_obj.ap_ is None and Z is None:
                    raise Exception('No data found in DS_obj')
                if DS_obj.ap_ is None:
                    warnings.warn('No added parameter data found in DS_obj. Defaulting to Z data',UserWarning)
                    self.Z_ = Z
                    self.tags_ = DS_obj.tags_
                elif Z is None:
                    warnings.warn('No Z data found in DS_obj. Defaulting to added parameter data',UserWarning)
                    self.Z_ = DS_obj.ap_
                    self.tags_ = DS_obj.ap_tags_
                else:
                    self.Z_ = np.hstack((Z,DS_obj.ap_))
                    self.tags_ = np.hstack((DS_obj.tags_,DS_obj.ap_tags_))
            else:
                raise Exception('data_select mode invalid')
                
        if isinstance(DS_obj,dtsp.EISdataset):
            self.data_type_ = 'EIS'
        elif isinstance(DS_obj,dtsp.DRTdataset):
            self.data_type_ = 'DRT'
        else:
            self.data_type_ = 'custom'
        self.params_ = DS_obj.params_var_
        self.params_names_ = DS_obj.params_names_
        self.params_units_ = DS_obj.params_units_
        self.dsID_ = DS_obj.dsID_
        self.contains_data_ = True
        self.data_selection_ = data_select
        self.dataname_ = DS_obj.name_

    def clear_data(self):
        self.Z_ = None
        if 'data_type_' in self.__dict__:
            del self.__dict__['data_type_']
        self.tags_ = None
        self.contains_data_ = False
        if 'data_selection_' in self.__dict__:
            del self.__dict__['data_selection_']
        if 'dataname_' in self.__dict__:
            del self.__dict__['dataname_']
        self.is_split_ = False
        self.is_scaled_ = False
        self.is_preprocessed_ = False
        if 'Z_train_' in self.__dict__:
            del self.__dict__['Z_train_']
        if 'Z_test_' in self.__dict__:
            del self.__dict__['Z_test_']
        if 'p_train_' in self.__dict__:
            del self.__dict__['p_train_']
        if 'p_test_' in self.__dict__:
            del self.__dict__['p_test_']
        if 'ID_train_' in self.__dict__:
            del self.__dict__['ID_train_']
        if 'ID_test_' in self.__dict__:
            del self.__dict__['ID_test_']
        if 'Z0_train_' in self.__dict__:
            del self.__dict__['Z0_train_']
        if 'Z0_test_' in self.__dict__:
            del self.__dict__['Z0_test_']
        
    def split_re_im(self, return_separate=False):
        iscmplx = np.iscomplex(self.Z_[0,:])
        if np.any(iscmplx):
            cmplx_idx = []
            for i in range(np.size(iscmplx)):
                if iscmplx[i]:
                    cmplx_idx.append(i)
            cmplx_idx = np.array(cmplx_idx)
            Zre = self.Z_[:,cmplx_idx].real
            Zim = self.Z_[:,cmplx_idx].imag
            Zadd = self.Z_[:,~iscmplx]
            tags = np.array(self.tags_)
            if not return_separate:
                if np.size(Zadd,axis=1) != 0:
                    self.Z_ = np.hstack((Zre,Zim,Zadd)).astype(float)
                    self.tags_ = np.hstack((['Re ' + t for t in tags[cmplx_idx]],['Im ' + t for t in tags[cmplx_idx]],tags[~cmplx_idx]))
                else:
                    self.Z_ = np.hstack((Zre,Zim)).astype(float)
                    self.tags_ = np.hstack((['Re ' + t for t in tags[cmplx_idx]],['Im ' + t for t in tags[cmplx_idx]]))
            else:
                return Zre, Zim, Zadd

    def vif(self):
        if not self.contains_data_:
            raise Exception('No data detected in  model')
        Zvif = add_constant(self.Z_)
        vif = []
        for i,_ in enumerate(self.Z_[0,:]):
            vif.append(variance_inflation_factor(Zvif,i))
        return np.array(vif)
    
    def train_test_split(self, test_size=0.2, random_state=None):
        '''
        TO DO:
            1. Add different modes (e.g., stratification)
            2. idk this seems incomplete

        Parameters
        ----------
        test_size : TYPE, optional
            DESCRIPTION. The default is 0.2.
        random_state : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''
        if not self.contains_data_:
            raise Exception('Data not loaded into model')
        self.Z_train_, self.Z_test_, self.p_train_, self.p_test_, self.ID_train_, self.ID_test_ = train_test_split(
            self.Z_,self.params_,self.dsID_,test_size=test_size,random_state=random_state
        )
        self.is_split_ = True
    
    def pyt_train_test_split(self, test_size=0.2, random_state=None):
        if not self.contains_data_:
            raise Exception('Data not loaded into model')
        Zre, Zim, Zadd = self.split_re_im(return_separate=True)
        Zadd = Zadd.astype(float)
        Zre_train, Zre_test, Zim_train, Zim_test, Zadd_train, Zadd_test, self.p_train_, self.p_test_, self.ID_train_, self.ID_test_ = train_test_split(
            Zre, Zim, Zadd, self.params_, self.dsID_,test_size=test_size,random_state=random_state    
        )
        Z_train = Zre_train + Zim_train*1j
        Z_test = Zre_test + Zim_test*1j
        self.Z_train_ = np.hstack((Z_train,Zadd_train))
        self.Z_test_ = np.hstack((Z_test,Zadd_test))
        self.is_split_ = True
        
    def scale_data(self, scaler=MaxAbsScaler(),ignore_infoleak=False):
        if not self.contains_data_:
            raise Exception('Data not loaded into model')
        if self.is_scaled_:
            warnings.warn('Scaled data detected; replacing old scaled data',UserWarning)
            if 'Z_train0_' in self.__dict__:
                self.Z_train_ = scaler.fit_transform(self.Z_train0_)
                self.Z_test_ = scaler.transform(self.Z_test0_)
            elif 'Z0_' in self.__dict__:
                self.Z_ = scaler.fit_transform(self.Z_)
        else:
            if self.is_split_:
                self.Z_train0_ = self.Z_train_.copy()
                self.Z_test0_ = self.Z_test_.copy()
                self.Z_train_ = scaler.fit_transform(self.Z_train_)
                self.Z_test_ = scaler.transform(self.Z_test_)
            else:
                if not ignore_infoleak:
                    raise Exception('Data not split into train and test sets. Scaling before splitting causes infoleak between sets. Set ignore_infoleak = True to proceed')
                else:
                    self.Z0_ = self.Z_.copy()
                    self.Z_ = scaler.fit_transform(self.Z_)
            self.is_scaled_ = True
        self.scaler_ = scaler
    
    def pyt_scale_data(self,scaler=MaxAbsScaler(),ignore_infoleak=False):
        '''
        TO DO:
            1. Add scaling for things other than magnitude (individual Re, Im components?)
            2. Doesn't support scaling already scaled data yet
        '''
        if not self.contains_data_:
            raise Exception('Data not loaded into model')
        if self.is_split_:
            Z_train_r, Z_train_phi, Z_test_r, Z_test_phi = complex_to_polar(self.Z_train_,self.Z_test_)
            Z_train_r = scaler.fit_transform(Z_train_r)
            Z_test_r = scaler.transform(Z_test_r)
            self.Z_train_, self.Z_test_ = polar_to_complex(Z_train_r,Z_train_phi,Z_test_r,Z_test_phi) 
        else:
            if not ignore_infoleak:
                raise Exception('Data not split into train and test sets. Scaling before splitting leaDS_obj to infoleak between sets. Set ignore_infoleak = True to proceed')
            else:
                raise NotImplementedError('Scaling for pyTorch models currently not supported for unsplit data')
                self.Z0_ = self.Z_.copy()
                self.Z_ = scaler.fit_transform(self.Z_)
        self.is_scaled_ = True
        self.scalar_ = scaler
        
    def rename(self, name):
        self.name_ = name
        
    def __getstate__(self):
        '''
        Returns a copy of the object dictionary.
        
        '''
        output = self.__dict__.copy()
        return output

'''
======================================
ML model analysis functions. These functions are primarily
concerned with comparing performance across models or datasets.
======================================
'''

def pyt_compare_models(model_list, processed_ModelDS,n_jobs=10,name_list = None,plot_comparison=True,
                       learning_rate = 1e-3, epochs=50, batch_size=100,dynamic_lr=False, dlr_epoch=30, 
                       dlr_scheduler='exp',scoring=True, score_method = 'default', verbose=True,**kwargs):
    losses = []
    if scoring:
        scores = []
    for m in model_list:
        if m is None:
            m0 = processed_ModelDS.model_.state_dict()
        score_m = np.zeros((n_jobs,epochs))
        loss_m = np.zeros((n_jobs,epochs))
        for n in range(n_jobs):
            if m is not None:
                processed_ModelDS.swap_model(m)
            else:
                processed_ModelDS.model_.load_state_dict(m0)
            pyt_train(processed_ModelDS, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, dynamic_lr=dynamic_lr, dlr_epoch=dlr_epoch, dlr_scheduler=dlr_scheduler,
              scoring=scoring,score_method=score_method,plotting=False,verbose=verbose,**kwargs)
            loss_m[n,:] = processed_ModelDS.loss_.flatten()
            if scoring:
                score_m[n,:] = processed_ModelDS.score_.flatten()
        losses.append(loss_m)
        scores.append(score_m)
    if plot_comparison:   
        if name_list is None:
            name_list = np.arange(1,len(model_list)+1,1).astype(str)
    if scoring:
        if plot_comparison:
            if score_method == 'default':
                if processed_ModelDS.model_type_ == 'regressor':
                    score_method = 'r2'
                if processed_ModelDS.model_type_ == 'classifier':
                    score_method = 'accuracy'
            plot_comparison(losses,scores,score_name=score_method,labels=name_list,plot_std=True)
        return losses, scores
    else:
        if plot_comparison:
            plot_comparison(losses,scores=None,score_name=score_method,labels=name_list,plot_std=True)
        return losses
    
#def pyt_compare_transformations(transformation_list, processed_ModelDS, other stuff yk)
