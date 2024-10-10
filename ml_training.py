import numpy as np
import dataset_processing as dtsp
import pytorch_integration as pyti
import plotgen
from analysis import complex_to_polar
from analysis import polar_to_complex
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
#from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

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

def pyt_preprocess_and_train(ModelDS, DS, name=None, cast_as_complex = False,learning_rate = 1e-3, epochs=50, batch_size=100, 
                             dynamic_lr = False, scoring=True, plotting=False,savefig=True,savepath='',verbose=True,**kwargs):
    if ModelDS.model_type_ != 'pyTorch Module':
        raise Exception('pytorch_train can only be used on pyTorch neural networks (subclasses of nn.Module)')
    
    data_select = kwargs.get('data_select','all')
    ModelDS.load_data(DS,data_select=data_select)
    if verbose:
        print('Data loaded\n-------------')
    test_size = kwargs.get('test_size',0.2)
    
    random_state = kwargs.get('random_state',None)
    if cast_as_complex: 
    # Complex support should be treated as a beta feature, as it is in pyTorch.
    # While the functionality is implemented here, and it works with certain custom Modules I've written,
    # these implementations currently rely on discarding the imaginary component or its loss gradient.
    # (which sorta defeats the whole purpose of wrapping both Zre and Zim into a single value,
    # as it's essentially unnecessary data loss when compared to split Re/Im float representations)
    # See: https://pytorch.org/docs/stable/complex_numbers.html
        ModelDS.is_complex_ = True
        ModelDS.pyt_train_test_split(test_size=test_size,random_state=random_state)
        if verbose:
            print('Data split into train/test sets\nTest size = {:.1%}\n-------------'.format(test_size))
        
        scaler = kwargs.get('scaler',MaxAbsScaler())
        ModelDS.pyt_scale_data(scaler=scaler)
        if verbose:
            print('Data scaled\n-------------')
            print('Training model\n-------------')
    else:
        ModelDS.is_complex_ = False
        ModelDS.split_re_im()
        ModelDS.train_test_split(test_size=test_size,random_state=random_state)
        if verbose:
            print('Data split into train/test sets\nTest size = {:.1%}\n-------------'.format(test_size))
        scaler = kwargs.get('scaler',MaxAbsScaler())
        ModelDS.scale_data(scaler=scaler)
        
        if verbose:
            print('Data scaled\n-------------')
            print('Training model\n-------------')
    
    pyt_train(ModelDS,learning_rate=learning_rate,epochs=epochs,dynamic_lr=dynamic_lr,batch_size=batch_size,
              scoring=scoring,plotting=plotting,verbose=verbose,**kwargs)
    
def pyt_train(ModelDS,learning_rate = 1e-3, epochs=50, batch_size=100,
              dynamic_lr=False, dlr_epoch= 30, dlr_scheduler='exp',
              scoring=True, plotting=False,verbose=True,**kwargs):
    '''
    TO DO:
        1. Handle different optimizers:
            1a. ISSUE: I don't think it's possible to pass optim.Optimizer directly, 
            as it would be initialized before the model is sent to device.
            1b. https://discuss.pytorch.org/t/should-i-create-optimizer-after-sending-the-model-to-gpu/133418
            It may be possible? Needs testing
        2.  [DONE] Handle different loss functions:
            2a. Should be relatively simple, as these can get passed explicitly and
            then sent to device.
            2b. BASE CLASS: _Loss(Module)
        3.  Handle schedulers better:
            3a. Users should be able to specify period of non-dynamic LR and switch to
            dynamic after t epochs.
            3b. Users should be able to pass any arbitrary scheduler

    Parameters
    ----------
    ModelDS : TYPE
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
    if ModelDS.model_type_ != 'pyTorch Module':
        raise Exception('pytorch_train can only be used on pyTorch neural networks (subclasses of nn.Module)')
    if ModelDS.is_complex_:
        dtypes = torch.cfloat
    else:
        dtypes = torch.float
    device = pyti.get_device()
    if verbose:
        print('Using {} device\n-------------'.format(device))
    model = ModelDS.model_.to(device,dtype=dtypes)
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
            target_weights = np.ones(np.size(ModelDS.p_train_,axis=1),1)
        elif isinstance(target_weights,str):
            if not target_weights == 'batch mean' or target_weights == 'none':
                raise Exception('Available loss_weights string inputs: batch mean or none')
        else:
            raise Exception('Invalid loss_weights input')
    else:
        target_weights = pyti.tensor_transform(np.mean(ModelDS.p_train_,axis=0),dtypes)
        
    training_data = pyti.Data(ModelDS.Z_train_,ModelDS.p_train_,dtypes=dtypes)
    testing_data = pyti.Data(ModelDS.Z_test_,ModelDS.p_test_,dtypes=dtypes)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

    if not dynamic_lr:
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)
        for t in range(epochs):
            if verbose:
                print(f"Epoch {t+1}\n-------------------------------")
            pyti.train_loop(train_dataloader,model,loss,optimizer,target_weights,batch_size=batch_size,dtypes=dtypes,verbose=verbose)
            pyti.test_loop(test_dataloader,model,loss,target_weights,dtypes=dtypes,verbose=verbose)
    else:
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)
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
            val_out = pyti.test_loop(test_dataloader, model, loss, target_weights)
            if dlr_scheduler != 'plateau':
                scheduler.step()
            elif dlr_scheduler == 'plateau':
                scheduler.step(val_out)
                current_lr = scheduler.get_last_lr()[0]
                if verbose:
                    if current_lr != old_lr:
                        print('New learning rate: {:.3e}'.format(current_lr))
                        old_lr = current_lr
            
    p_test_pred = model(torch.from_numpy(ModelDS.Z_test_).to(device,dtype=dtypes))
    p_test_pred = p_test_pred.detach().cpu().numpy()
    p_train_pred = model(torch.from_numpy(ModelDS.Z_train_).to(device,dtype=dtypes))
    p_train_pred = p_train_pred.detach().cpu().numpy()
    
    ModelDS.p_test_pred_ = p_test_pred
    ModelDS.p_train_pred_ = p_train_pred
    del p_test_pred
    del p_train_pred
    
    if scoring:
        get_scores(ModelDS,verbose=verbose)
    if plotting:
        titles = kwargs.get('titles',ModelDS.params_names_)
        axes_units = kwargs.get('axes_units',ModelDS.params_units_)
        plotgen.plot_results(titles,axes_units,ModelDS.p_test_,ModelDS.p_test_pred_,ModelDS.name_,ModelDS.dataname_)

'''
======================================
Scikit-learn preprocessing and training
======================================
'''

def preprocess_and_train(ModelDS, DS, name=None, scoring=True, plotting=False,
                         savefig=True,savepath='',verbose=True,**kwargs):
    if ModelDS.model_type_ != 'Random Forest' and  ModelDS.model_type_ != 'Multi-layer Perceptron':
        raise Exception('preprocess_and_train() only valid for sk-learn models. Currently supported: Random Forest, Multi-layer Perceptron')
    
    data_select = kwargs.get('data_select','all')
    ModelDS.load_data(DS,data_select=data_select)
    ModelDS.split_re_im()
    if verbose:
        print('Data loaded and split into real and imaginary components\n-------------')
    test_size = kwargs.get('test_size',0.2)
    random_state = kwargs.get('random_state',None)
    ModelDS.train_test_split(test_size=test_size,random_state=random_state)
    if verbose:
        print('Data split into train/test sets\nTest size = {:.1%}\n-------------'.format(test_size))
    
    scaler = kwargs.get('scaler',MaxAbsScaler())
    ModelDS.scale_data(scaler=scaler)
    if verbose:
        print('Data scaled\n-------------')
        print('Training model\n-------------')
    train(ModelDS,scoring=scoring,plotting=plotting,verbose=verbose)
        
def train(ModelDS,scoring=True,plotting=False,verbose=True,**kwargs):
    if ModelDS.model_type_ != 'Random Forest' and  ModelDS.model_type_ != 'Multi-layer Perceptron':
        raise Exception('train() only valid for sk-learn models. Currently supported: Random Forest, Multi-layer Perceptron')
    
    Z_train, Z_test, p_train = ModelDS.Z_train_, ModelDS.Z_test_, ModelDS.p_train_
    ModelDS.model_.fit(Z_train, p_train)
    p_train_pred = ModelDS.model_.predict(Z_train)
    p_test_pred = ModelDS.model_.predict(Z_test)
    ModelDS.p_train_pred_ = p_train_pred
    ModelDS.p_test_pred_ = p_test_pred
    
    if scoring:
        get_scores(ModelDS,verbose=verbose)
    if plotting:
        titles = kwargs.get('titles',ModelDS.params_names_)
        axes_units = kwargs.get('axes_units',ModelDS.params_units_)
        plotgen.plot_results(titles,axes_units,ModelDS.p_test_,ModelDS.p_test_pred_,ModelDS.name_,ModelDS.dataname_)
        
def get_scores(ModelDS,verbose=True):
    # ONLY WORKS FOR REGRESSORS REALLY
    r2_test = metrics.r2_score(ModelDS.p_test_,ModelDS.p_test_pred_,multioutput='raw_values')
    r2_train = metrics.r2_score(ModelDS.p_train_,ModelDS.p_train_pred_,multioutput='raw_values')
    rmse_test = metrics.root_mean_squared_error(ModelDS.p_test_,ModelDS.p_test_pred_,multioutput='raw_values')
    rmse_train = metrics.root_mean_squared_error(ModelDS.p_train_,ModelDS.p_train_pred_,multioutput='raw_values')
    if verbose:
        for x in range(np.size(r2_test)):
            print('{}:\nR2 test: {:.3f}\nR2 train: {:.3f}\nRMSE test: {:.2e}\nRMSE train{:.2e}\n-------------'.format(ModelDS.params_names_[x],r2_test[x],r2_train[x],rmse_test[x],rmse_train[x]))
    ModelDS.r2_test_, ModelDS.r2_train_, ModelDS.rmse_test_, ModelDS.rmse_train_ = r2_test, r2_train, rmse_test, rmse_train

'''
======================================
Combined model-dataset class used as the basis of storing, 
training, and evaluating model-dataset pairs.
======================================
'''
    
class ModelDS():
    '''
    TO DO:
        1. Figure out how to dynamically find the model type and create self.model_type variable from that
    '''
    def __init__(self, model, name=None):
        if isinstance(model,RandomForestRegressor) or isinstance(model,RandomForestClassifier):
            self.model_type_ = 'Random Forest'
        elif isinstance(model,MLPRegressor) or isinstance(model,MLPClassifier):
            self.model_type_ = 'Multi-layer Perceptron'
        elif isinstance(model, nn.Module):
            self.model_type_ = 'pyTorch Module'
        else:
            raise NotImplementedError('Custom models not supported yet :(')
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
    
    def swap_model(self,model,name=None):
        if isinstance(model,RandomForestRegressor) or isinstance(model,RandomForestClassifier):
            self.model_type_ = 'Random Forest'
        elif isinstance(model,MLPRegressor) or isinstance(model,MLPClassifier):
            self.model_type_ = 'Multi-layer Perceptron'
        elif isinstance(model, nn.Module):
            self.model_type_ = 'pyTorch Module'
        else:
            raise NotImplementedError('Custom models not supported yet :(')
        if name is None:
            self.name_ = self.model_type_
        else:
            self.name_ = name
        self.model_ = model
    
    def load_data(self,DS, data_select = 'all'):
        # NEED TO CHANGE THIS LOGIC -> IF data_select == 'all' but one thing not provided,
        # default to loading data that is present.
        
        # ALSO GRAB PARAMS
        if isinstance(DS,dtsp.EISdataset):
            if self.contains_data_:
                raise Exception('Data already found; run Model.clear_data before loading again')
            if data_select == 'noadded':
                if DS.Z_var_ is None:
                    raise Exception('No Z data found in DS')
                self.Z_ = DS.Z_var_
                self.tags_ = DS.tags_
            elif data_select == 'added':
                if DS.ap_ is None:
                    raise Exception('No added parameters found in DS')
                self.Z_ = DS.ap_
                self.tags_ = DS.ap_tags_
            elif data_select == 'all':
                if DS.ap_ is None and DS.Z_var_ is None:
                    raise Exception('No data found in DS')
                if DS.ap_ is None:
                    warnings.warn('No added parameter data found in DS. Defaulting to Z data',UserWarning)
                    self.Z_ = DS.Z_var_
                    self.tags_ = DS.tags_
                elif DS.Z_var_ is None:
                    warnings.warn('No Z data found in DS. Defaulting to added parameter data',UserWarning)
                    self.Z_ = DS.ap_
                    self.tags_ = DS.ap_tags_
                else:
                    self.Z_ = np.hstack((DS.Z_var_,DS.ap_))
                    print(self.Z_.shape)
                    self.tags_ = np.hstack((DS.tags_,DS.ap_tags_))
            else:
                raise Exception('data_select mode invalid')
            self.data_type_ = 'EIS'

        elif isinstance(DS,dtsp.DRTdataset):
            if self.contains_data_:
                raise Exception('Data already found; run [MODEL].clear_data before loading again')
            if data_select == 'noadded':
                if DS.DRTdata_ is None:
                    raise Exception('No DRT data found in DS')
                self.Z_ = DS.DRTdata_
                self.tags_ = DS.tags_
            elif data_select == 'added':
                if DS.ap_ is None:
                    raise Exception('No added parameters found in DS')
                self.Z_ = DS.ap_
                self.tags_ = DS.ap_tags_
            elif data_select == 'all':
                if DS.ap_ is None:
                    raise Exception('No added parameters found in DS')
                if DS.DRTdata_ is None:
                    raise Exception('No DRT data found in DS')
                self.Z_ = np.hstack((DS.DRTdata_,DS.ap_))
                self.tags_ = np.hstack((DS.tags_,DS.ap_tags_))
            self.data_type_ = 'DRT'
        
        self.params_ = DS.params_var_
        self.params_names_ = DS.params_names_
        self.params_units_ = DS.params_units_
        self.circuitID_ = DS.circuitID_
        self.contains_data_ = True
        self.data_selection_ = data_select
        self.dataname_ = DS.name_

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
            self.Z_,self.params_,self.circuitID_,test_size=test_size,random_state=random_state
        )
        self.is_split_ = True
    
    def pyt_train_test_split(self, test_size=0.2, random_state=None):
        if not self.contains_data_:
            raise Exception('Data not loaded into model')
        Zre, Zim, Zadd = self.split_re_im(return_separate=True)
        Zadd = Zadd.astype(float)
        Zre_train, Zre_test, Zim_train, Zim_test, Zadd_train, Zadd_test, self.p_train_, self.p_test_, self.ID_train_, self.ID_test_ = train_test_split(
            Zre, Zim, Zadd, self.params_, self.circuitID_,test_size=test_size,random_state=random_state    
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
                    raise Exception('Data not split into train and test sets. Scaling before splitting leads to infoleak between sets. Set ignore_infoleak = True to proceed')
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
                raise Exception('Data not split into train and test sets. Scaling before splitting leads to infoleak between sets. Set ignore_infoleak = True to proceed')
            else:
                raise NotImplementedError('Scaling for pyTorch models currently not supported for unsplit data')
                self.Z0_ = self.Z_.copy()
                self.Z_ = scaler.fit_transform(self.Z_)
        self.is_scaled_ = True
        self.scalar_ = scaler
        
    def update_name(self, name):
        self.name_ = name
        
    def __getstate__(self):
        '''
        Returns a copy of the object dictionary.
        
        '''
        output = self.__dict__.copy()
        return output