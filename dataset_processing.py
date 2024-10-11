'''
Jack Pereira
Department of Chemical Engineering
University of Washington
2024
'''
import os
import time
import warnings
import datetime
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from impedance import preprocessing
from impedance.models.circuits.fitting import circuit_fit
from impedance.models.circuits import CustomCircuit
from impedance.visualization import plot_nyquist
from impedance import validation
from sklearn import metrics
import DRT_main as DRT
import analysis

'''
======================================
Generic read, save, and load functions.
======================================
'''

def read_files(path):
    """ Function
    TO DO: 
        1. HANDLE FILE TYPES OTHER THAN .Z
        2. CREATE EXCEPTIONS FOR NON-UNIFORM FREQUENCY RANGES
        3. CREATE EXCEPTIONS FOR NO FILES FOUND
    Parameters
    ----------
    path : string
      
    Returns
    -------
    mu : float
        under- or over-fitting measure from linKK
    
    M : int
        number of RC elements used in linKK
    
    params : numpy array
        fitted parameter array    
    """
    if path is None:
        Z_dict = None
        f_dict = None
        Z_files = None
    else:
        dir_list = os.listdir(path)
        dir_list = [d for d in dir_list if os.path.isfile(os.path.join(path,d))]
        Z_files = [z for z in dir_list if z.endswith('.z')]
        filenames = [os.path.splitext(f)[0] for f in Z_files]
        
        Z_dict = {}
        f_dict = {}
        
        for i,file in enumerate(filenames):
                f, Z = preprocessing.readZPlot(path+Z_files[i])
                Z_dict.update({file: Z})
                f_dict.update({file: f})
    
    return f_dict, Z_dict, Z_files

def save_obj(saved_object, to_file = True, to_var = False, savepath = ''):
    name = saved_object.name_
    stream = pickle.dumps(saved_object)
    if to_file:
        filename = savepath+name+'.pickle'
        with open(f"{filename}",'wb') as output_file:
            pickle.dump(saved_object, output_file, pickle.HIGHEST_PROTOCOL)
    if to_var:
        return stream
    
def load_obj(save_location):
    if isinstance(save_location,bytes):
        output = pickle.loads(save_location)
    elif isinstance(save_location,str):
        with open(save_location,'rb') as input_file:
            output = pickle.load(input_file)
    else:
        raise Exception('Invalid save location provided')
    return output

'''
======================================
Dataset superclass. 
Contains functions for saving, renaming, getting parameters, etc.
======================================
'''

class DS():
    def __init__():
        return None
    
    def addparams(self, added_array, tags = None):
        '''
        Generic function to store additional parameters alongside impedance data.
    
        Parameters
        ----------
        added_array : numpy.ndarray
            Array of additional parameters to store alongside core impedance dataset.
        tags : numpy.ndarray,dtype=str or list,dtype=str, optional
            Descriptions of each column in added_array 
    
        '''
        if self.ap_ is None:
            if tags is None:
                tags = np.array([str('ID ' + x) for x in np.arange(1,np.size(added_array,axis=1))])
            elif isinstance(tags,list):
                tags = np.array(tags)
            if not isinstance(tags,np.ndarray):
                raise Exception('tags must be a numpy array or list')
            self.ap_ = added_array
            self.ap_tags_ = tags
        else:
            if tags is None:
                nparams_existing = np.size(self.ap_)
                tags = np.array([str('ID ' + x) for x in np.arange(nparams_existing,np.size(added_array,axis=1)+nparams_existing)])
            elif isinstance(tags,list):
                tags = np.array(tags)
            if not isinstance(tags,np.ndarray):
                raise Exception('tags must be a numpy array or list')
            self.ap_ = np.hstack(self.ap_,added_array)
            self.ap_tags_ = np.hstack(self.ap_tags_,tags)
            
    def rename(self, name):
        '''
        Updates name attribute.
        
        '''
        self.name_ = name
        
    def clear_addparams(self):
        '''
        Clears the additional parameter and corresponding tag attributes.
    
        '''
        self.ap_ = None
        self.ap_tags_ = None
    
    def get_data(self):
        '''
        Returns the EIS data array and corresponding frequency (1/s) values.      
    
        '''
        return self.Z_var_, self.ap_, self.f_gen_
    
    def get_tags(self):
        '''
        Returns tagged labels for the core dataset and any additional parameters.
    
        '''
        return self.tags_, self.ap_tags_
    
    def __getstate__(self):
        '''
        Returns a copy of the object dictionary.
        
        '''
        output = self.__dict__.copy()
        return output

'''
======================================
EIS dataset Class, with built-in parameter estimation 
and variation generation functions.
======================================
'''

class EISdataset(DS):
    def __init__(self, path, dataset_name, from_files = True):
        self.name_ = dataset_name
        if from_files:
            f, Z, files = read_files(path)
            self.f_dir_ = f
            self.Z_dir_ = Z
            self.files_ = files 
            if self.f_dir_ == {}:
                warnings.warn('No valid files found in filepath. Base data is currently empty',UserWarning)
                self.source_ = 'none'
                self.datatag_ = 'none'
            else:
                self.source_ = 'Z files'
                self.datatag_ = 'standard'
        else:
            self.f_dir_ = {}
            self.Z_dir_ = {}
            self.files_ = []
        self.is_fit_ = False
        self.Z_var_ = None
        self.tags_ = None
        self.params_var_ = None
        self.ap_ = None
        self.ap_tags_ = None
    
    # change to class method!
    def from_npy(self,featurepath,targetpath, f_gen, additional_params = 0):
        Z = np.load(featurepath)
        if np.any(np.issubdtype(Z[0].dtype, np.complexfloating)):
            self.Z_var_ = Z[:,:-additional_params]
            self.ap_ = Z[:,-additional_params:] # CHECK VALIDITY OF THIS
        elif np.any(np.issubdtype(Z[0].dtype, np.floating)):
            # CHECK TO SEE IF THE Z DATA IS EVEN IN SIZE, OTHERWISE RAISE EXCEPTION
            hidx = int(np.size(Z[0]-additional_params)/2)
            if np.mean(Z[0][:hidx]) > np.mean(Z[0][hidx:]):
                self.Z_var_ = Z[:,:hidx] + Z[:,hidx:2*hidx]*1j
                self.ap_ = Z[:,:-2*hidx]
            else:
                self.Z_var_ = Z[:,hidx:2*hidx] + Z[:,:hidx]*1j
                self.ap_ = Z[:,-2*hidx:]
        else:
            raise Exception('Data cannot be read. Ensure it contains complex floating or floating dtypes')
        self.params_var_ = np.load(targetpath)
        self.feature_data_ = featurepath
        self.target_data_ = targetpath
        self.f_gen_ = f_gen
        self.source_ = 'npy file'
        self.tags_ = np.char.mod('%.2e', f_gen)
        
    def get_circuit_parameters(self, circuit_string, parameter_guess, f_gen = np.logspace(5,-1,61), global_optimization=False,
                               plot=True, save_figure=False, linKK=True,
                               ignore_negative_data=False, validate=True, weight_by_modulus=True, verbose=True,
                               **kwargs):
        '''
        TO DO:
            1. Add verbose printout statements
            2. Handle parameter_guess being 1 or multiple sets
            3. Handle multiple possible circuit strings for ideal fitting
            4. Handle |Z| weighting
            
        Parameters
        ----------
        circuit_string : str
            Circuit string defining the equivalent circuit to use for fitting.
            
        parameter_guess : TYPE
            DESCRIPTION.
        global_optimization : TYPE, optional
            DESCRIPTION. The default is False.
        generate_plot : TYPE, optional
            DESCRIPTION. The default is True.
        save_figure : TYPE, optional
            DESCRIPTION. The default is False.
        linKK : TYPE, optional
            DESCRIPTION. The default is True.
        ignore_negative_data : TYPE, optional
            DESCRIPTION. The default is True.
        validate : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        '''
        
        ndata = len(self.files_)
        nfreq = len(f_gen)
        nparams = len(parameter_guess)
        self.circuit_string_ = circuit_string
        self.params_ = np.empty((ndata,nparams))
        self.Z_basefit_ = np.empty((ndata,nfreq),dtype=complex)
        'ADD SOMETHING HERE TO HANDLE PARAMETER_GUESS BEING 1 OR MULTIPLE SETS'
        
        filenames = [os.path.splitext(f)[0] for f in self.files_]
        
        if validate:
            self.M_ = np.empty((ndata,1))
            self.mu_ = np.empty((ndata,1))
            self.Z_linKK_ = {}
            self.real_residual_linKK_ = {}
            self.imag_residual_linKK_ = {}
            #self.Z_linKK = np.empty((ndata,nfreq),dtype=complex)
            #self.real_residual_linKK = np.empty((ndata,nfreq))
            #self.imag_residual_linKK = np.empty((ndata,nfreq))
            
        for i in range(ndata):
            f_exp = self.f_dir_[filenames[i]]
            Z = self.Z_dir_[filenames[i]]
            if 'freq_limits' in kwargs:
                f_lim = kwargs.get('freq_limits')
                f_exp, Z = preprocessing.cropFrequencies(f_exp,Z,f_lim[0],f_lim[1])
            elif ignore_negative_data == True:
                f_exp, Z = preprocessing.ignoreBelowX(f_exp,Z)
                
            p_fit, p_error = circuit_fit(f_exp, Z, circuit_string, parameter_guess, 
                                        global_opt=global_optimization)
            circuit = CustomCircuit(circuit=circuit_string,initial_guess=p_fit)
            circuit.parameters_ = p_fit
            circuit.conf_ = p_error
            Z_basefit_i = circuit.predict(f_gen)
            self.Z_basefit_[i,:] = Z_basefit_i
            self.params_[i,:] = p_fit
            
            if validate:
                M, mu, Z_linKK, real_residual_linKK, imag_residual_linKK = validation.linKK(f_exp,Z)
                self.M_[i] = M
                self.mu_[i] = mu
                self.Z_linKK_.update({filenames[i]: Z_linKK})
                self.real_residual_linKK_.update({filenames[i]: real_residual_linKK})
                self.imag_residual_linKK_.update({filenames[i]: imag_residual_linKK})
            if plot:
                fix,ax = plt.subplots(figsize=(10,8))
                ax = plot_nyquist(Z,ax=ax,c='black',label='Experimental Data')
                ax = plot_nyquist(Z_basefit_i,fmt='-',ax=ax,c='blue',label='Circuit Fit')
                if validate:
                    ax = plot_nyquist(Z_linKK,fmt='-',ax=ax,c='red',label='LinKK:\nM = {}\n$\\mu$ = {:.3f}'.format(M,mu),alpha=0.7)
                plt.legend(loc='upper left')
                plt.title(filenames[i])
                plt.tight_layout()
                if save_figure:
                    filepath = kwargs.get('filepath','')
                    dpi = kwargs.get('dpi',800)
                    plt.savefig(filepath+filenames[i],dpi=dpi)
                plt.show()
        self.f_gen_ = f_gen
        self.is_fit_ = True
        self.params_names_ = circuit.get_param_names()[0]
        self.params_units_ = circuit.get_param_names()[1]
        self.tags_ = np.char.mod('%.2e',f_gen)
        
        if verbose:
            print('Circuit fitting complete!')
    
    def test_circuits(self,circuit_strings,initial_guesses,score='MAPE',names=None,global_optimization=False,validate=True,save_figure=False,**kwargs):
        ndata = len(self.files_)
        ncir = len(circuit_strings)
        filenames = [os.path.splitext(f)[0] for f in self.files_]
        if names is None:
            names = circuit_strings
        if score is not None:
            scores = np.empty((ncir,ndata))
        
        for i in range(ndata):
            f_exp = self.f_dir_[filenames[i]]
            Z_exp = self.Z_dir_[filenames[i]]
            Z_pred = np.empty((ncir,np.size(Z_exp)),dtype=complex)
            fig,ax = plt.subplots(figsize=(10,8))
            ax = plot_nyquist(Z_exp,fmt='-',ax=ax,c='blue',label='Circuit Fit')
            for c in range(ncir):
                p_fit, p_error = circuit_fit(f_exp, Z_exp, circuit_strings[c], initial_guesses[c], 
                                            global_opt=global_optimization)
                circuit = CustomCircuit(circuit=circuit_strings[c],initial_guess=p_fit)
                circuit.parameters_ = p_fit
                circuit.conf_ = p_error
                Z_pred[c,:] = circuit.predict(f_exp)
                ax = plot_nyquist(Z_pred[c,:], ax=ax,label=names[c])
                if score == 'MAPE':
                    scores[c,i] = metrics.mean_absolute_percentage_error(abs(Z_exp), abs(Z_pred[c]))
                elif score == 'MSE':
                    scores[c,i] = metrics.mean_squared_error(abs(Z_exp),abs(Z_pred[c]))
                else:
                    raise NotImplementedError('Other scoring metrics will be implemented in the future.')
            if validate:
                M, mu, Z_linKK, real_residual_linKK, imag_residual_linKK = validation.linKK(f_exp,Z_exp)
                ax = plot_nyquist(Z_linKK,fmt='-',ax=ax,c='red',label='LinKK:\nM = {}\n$\\mu$ = {:.3f}'.format(M,mu),alpha=0.7)
            if save_figure:
                filepath = kwargs.get('filepath','')
                dpi = kwargs.get('dpi',800)
                plt.savefig(filepath+filenames[i],dpi=dpi)
            plt.legend(loc='upper left')
            plt.title(filenames[i])
            plt.tight_layout()
            plt.show()
            
        scores = pd.DataFrame(data=scores,index=circuit_strings,columns=filenames)
        return scores
    
    def generate_variation(self, nvar_per_circuit, var_bounds, bound_type='relative', var_type='per circuit',difference=False, verbose=True):
        """
        TO DO:
            1. Check to make sure all self. parameters are updated
            2. Handle other types of generation other than uniform

        Parameters
        ----------
        nvar_per_circuit : TYPE
            DESCRIPTION.
        var_bounds : TYPE
            DESCRIPTION.
        absolute_bounds : TYPE, optional
            DESCRIPTION. The default is False.
        var_type : TYPE, optional
            DESCRIPTION. The default is 'uniform'.
        difference : TYPE, optional
            DESCRIPTION. The default is False.
        verbose : TYPE, optional
            DESCRIPTION. The default is True.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if not self.is_fit_:
            raise Exception('Dataset not fit to circuit parameters. Run object.get_circuit_parameters before generating variations.')
            
        nparams = np.size(self.params_,axis=1)
        ndata = len(self.files_)
        
        if isinstance(var_bounds,list):
            var_bounds = np.array(var_bounds)
        if var_type == 'per circuit':
            single_bound = False
            uniform_bounds = False
            if bound_type == 'absolute':
                if np.size(var_bounds,axis=1) != nparams or np.size(var_bounds,axis=0) != ndata:
                    raise Exception('Variation bounds not properly set. See documentation for valid inputs.')
                elif len(var_bounds) != nparams and len(var_bounds) != 1:
                    raise Exception('Variation bounds not properly set. See documentation for valid inputs.')
                elif len(var_bounds) == nparams:
                    uniform_bounds = True
                elif len(var_bounds) == 1:
                    single_bound = True
                else:
                    raise NotImplementedError('Complete definition of per-circuit + per-element bounds not implemented yet')
            elif bound_type == 'relative':
                if len(var_bounds) != nparams and len(var_bounds) != 1:
                    raise Exception('Variation bounds not properly set. See documentation for valid inputs.')
                elif len(var_bounds) == nparams:
                    uniform_bounds = True
                elif len(var_bounds) == 1:
                    single_bound = True
            else:
                raise Exception('Invalid bound_type entry')
        elif var_type == 'dataset':
            if bound_type == 'absolute':
                if np.size(var_bounds) != nparams:
                    raise Exception('Variation bounds not properly set. See documentation for valid inputs.')
        if verbose:
            if difference:
                print('Generating Z data with difference from baseline circuit adjustment\n=============')
            else:
                print('Generating Z data without difference from baseline adjustment\n=============')
                
        gen = np.random.Generator(np.random.PCG64())
        var_circuit = CustomCircuit(circuit=self.circuit_string_, initial_guess=self.params_[0,:])
        nfreq = len(self.f_gen_)
        nvar = int(ndata*nvar_per_circuit)
        
        self.Z_var_ = np.empty((nvar+ndata,nfreq),dtype=complex)
        if not difference:
            self.Z_var_[:ndata,:] = self.Z_basefit_
        else:
            self.Z_var_[:ndata,:] = 0
        self.params_var_ = np.empty((nvar+ndata,nparams))
        self.params_var_[:ndata,:] = self.params_
        self.circuitID_ = np.empty((nvar+ndata,1))
        self.circuitID_[:ndata] = np.arange(0,ndata,1).reshape(-1,1)
        
        total_var = ndata
        if var_type == 'dataset':
            if bound_type == 'absolute':
                lowerbound = var_bounds[:][0]
                upperbound = var_bounds[:][1]
            elif bound_type == 'relative':
                lowerbound = np.array([l[0] for l in var_bounds])*np.min(self.params_,axis=0)
                upperbound = np.array([u[1] for u in var_bounds])*np.max(self.params_,axis=0)
            else:
                raise Exception('Invalid bound_type value')
                
        for c in range(ndata):
            current_var = 0
            current_params = self.params_[c,:]
            if verbose:
                if self.files_ != []:
                    print('{} Base Parameters\n-------------'.format(self.files_[c]))
                else:
                    print('Circuit {} Base Parameters\n-------------'.format([c]))
                for p in range(nparams):
                    print('{} ({}): {:.3e}'.format(self.params_names_[p],self.params_units_[p],current_params[p]))
                print('\n-------------')
            while current_var < nvar_per_circuit:
                if var_type == 'per circuit':
                    if bound_type == 'relative':
                        if uniform_bounds:
                            lowerbound = current_params*(var_bounds[:][0][0])
                            upperbound = current_params*(var_bounds[:][0][1])
                        elif single_bound:
                            lowerbound = current_params*(var_bounds[0][0])
                            upperbound = current_params*(var_bounds[0][1])
                        else:
                            lowerbound = current_params*(var_bounds[c][:][0])
                            upperbound = current_params*(var_bounds[c][:][1])
                    else:
                        lowerbound = var_bounds[c][:][0]
                        upperbound = var_bounds[c][:][1]
                #if var_type == 'uniform': # changing var_type to new variable; var_type describes whether
                # variations are per-circuit or for the whole dataset
                var_params = gen.uniform(lowerbound,upperbound)
                #else:
                    #raise NotImplementedError('I need to add other generation forms')
                var_circuit.parameters_ = var_params
                
                if not difference:
                    self.Z_var_[total_var,:] = var_circuit.predict(self.f_gen_)
                else:
                    self.Z_var_[total_var,:] = self.Z_basefit_[c,:] - var_circuit.predict(self.f_gen_)
                
                self.params_var_[total_var,:] = var_params
                self.circuitID_[total_var] = c
                current_var += 1
                total_var += 1
            if verbose:
                print('Progress: [{}/{}]'.format(total_var,nvar+ndata))
                
        if verbose:
            print('Random variation complete!')
        self.source_ = 'generated' 

    def separate_real_imag(self):
        # Maybe delete this
        Z_re = np.real(self.Z_var_)
        Z_im = np.imag(self.Z_var_)
        return Z_re, Z_im
    

'''
======================================
DRT dataset Class, with built-in function to 
transform EIS data into equivalent DRT spectra.
======================================
'''
        
class DRTdataset(DS):
    def __init__(self,EISdata):
        self.f_gen_ = EISdata.f_gen_
        self.circuitID_ = EISdata.circuitID_
        self.params_var_ = EISdata.params_var_
        self.params_names_ = EISdata.params_names_
        self.params_units_ = EISdata.params_units_
        self.contains_data_ = False
        self.name_ = EISdata.name_ + 'DRT'
        self.DRTdata_ = None
        self.tags_ = None
        self.ap_ = None
        self.ap_tags_ = None

    def drt_transform(self,EISdata,fine=True, timer = True, timer_frac = 0.05, **kwargs):
        '''
        Transforms EIS data into a corresponding DRT dataset.

        Parameters
        ----------
        EISdata : EISdataset
            EIS data to be transformed (specifically, EISdata.Z_var_ is transformed).
            Transformation on baseline data will be implemented in the future.
        fine : bool, optional
            By default, the pyDRTtools transform returns 10*(number of frequencies) tau and gamma values.
            If True, save all values generated. If False, save (number of frequencies) instead. 
            Can be useful for reducing object datasize, especially for large datasets. 
            Since all values are generated either way, this has minimal impact on computational time.
            NOTE: In my experience, models trained on fine datasets display slightly improved performance 
            when compared to their course equivalents. However, the additional memory usage and computational costs
            during training may outweigh these benefits.
        timer : bool, optional
            If True, print periodic time estimation and progress updates.
        timer_frac : float in range (0,1), optional
            Sets the frequency of timer print statements as a fraction of total dataset.
        **kwargs : TYPE
            All **kwargs used in DRT_main.Simple_run() from pyDRTtools.

        '''
        DRT_obj = DRT.EIS_object(self.f_gen_,1,1)
        ndata = np.size(EISdata.Z_var_,axis=0)
        if fine:
            self.tau_ = DRT_obj.tau_fine
        else:
            self.tau_ = DRT_obj.tau
            self.DRTdata_ = np.empty((ndata,len(self.tau_)))
            self.tags_ = np.char.mod('$.2e', self.tau_fine_)
            
        rbf_type = kwargs.get('rbf_type','Gaussian')
        data_used = kwargs.get('data_used','Combined Re-Im Data')
        induct_used = kwargs.get('induct_used',1)
        der_used = kwargs.get('der_used','2nd order')
        lambda_value = kwargs.get('lambda_value',1E-3)
        shape_control = kwargs.get('shape_control','FWHM Coefficient')
        coeff = kwargs.get('coeff',0.5)
        
        if timer:
            timed_arr = np.arange(timer_frac,1,timer_frac)
            timed_iter = np.ceil(timed_arr*ndata).astype(int)
            time_counter = 0
            t_start = time.time()
        
        for i in range(ndata):
                
            DRT_obj.Z_exp = EISdata.Z_var_[i,:]
            DRT.Simple_run(DRT_obj, rbf_type, data_used, induct_used,
                           der_used, lambda_value, shape_control, coeff)
            if fine:
                self.DRTdata_[i,:] = DRT_obj.gamma
            else:
                self.DRTdata_[i,:] = DRT_obj.gamma[::10]

            if timer:
                if i in timed_iter:
                    t_end = time.time()
                    s = (t_end - t_start)*(1/timed_arr[time_counter]-1)
                    time_remaining = str(datetime.timedelta(seconds=round(s)))
                    print('DRT sets generated: [{}/{}]'.format(i,ndata))
                    print("Estimated time remaining: "+time_remaining+'\n-------------')
                    time_counter+=1
                    
        self.contains_data_ = True
        
    def add_highf_real(self,EISdata):
        '''
        Adds the highest frequency real Z datapoint as an additional parameter. This is generally a good idea for training 
        ML models on DRT datasets when an ohmic resistance is present, as DRT does a poor job of identifying R_ohm on its own.
        
        Parameters
        ----------
        EISdata : EISObject()
            Object containing loaded EIS data

        '''
        if EISdata.Z_var_ is None:
            raise Exception('No data found in EIS object')
        highf_real = np.array(np.real(EISdata.Z_var_[:,0]))
        highf_real = highf_real[:,np.newaxis]
        if self.ap_ is None:
            self.ap_ = highf_real
            self.ap_tags_ = np.array(['highf real'])
        else:
            if np.isin('high f Re', self.ap_tags_):
                raise Exception('High frequency real component already added')
            self.ap_ = np.hstack((self.ap_,highf_real))
            self.ap_tags_ = np.hstack((self.ap_tags_,np.array(['high f Re'])))
    
    def peak_analysis(self, n_peaks, cutoff_frac = 0.1, get_gamma = True, get_tau = True, get_area = True,
                       verbose = True, der = 1):
        
        peakdata, tags = analysis.find_drt_peaks(self, n_peaks, cutoff_frac=cutoff_frac, get_gamma=get_gamma,get_tau=get_tau,
                                                 get_area=get_area,separate_returns=False,print_stats=verbose,der=der
        )
        tags = np.array(tags)
        if self.ap_ is None:
            self.ap_ = peakdata
        else:
            if np.isin('Peak 1 top gamma', self.ap_tags_) or np.isin('Peak 1 top gamma', self.ap_tags_) or np.isin('Peak 1 area', self.ap_tags_):
                raise Exception('Peak data already added')
            self.ap_ = np.hstack((self.ap_,peakdata))
            self.ap_tags_ = np.hstack((self.ap_tags_,tags))
    
    
'''
======================================
Transformer class, subclasseed to store custom
user parameter transformations.
======================================
'''
class Transformer():
    '''
    I'm working on making proper documentation for usage here.
    The gist, for now, is that users should make a subclass containing the following:
        1. transform(self,params) that transforms parameters and returns new p, names, and units as lists
        2. (optional) inv_transform(self,params) that reverses transform()
    Example below:
    -----------------
class omega_transform(Transformer):
    def __init__(self):
        return None
    
    def transform(self,params,constants=None):
        # transforms EQC parameters for the circuit 'R0-p(R2-G2,C2)' into impedance base units (1/s, R)
        rohm = params[:,0]
        r2 = params[:,1]
        w2 = 1/(params[:,1]*params[:,4])
        rg = params[:,2]
        wg = 1/params[:,3]
        
        params_trns = [rohm,r2,w2,rg,wg]
        names_trns = ['R0','R2','W2','Rg','Wg']
        units_trns = ['Ohm','Ohm','1/s','Ohm','1/s']

        return params_trns, names_trns, units_trns
    
    def inv_transform(self,params_constants=None):
        rohm = params[:,0]
        r2 = params[:,1]
        rg = params[:,3]
        tg = params[:,3]/params[:,4]
        c2 = params[:,1]/params[:,2]
        
        params_inv = [rohm,r2,rg,tg,c2]
        names_inv = ['R0','R2','Rg','tg','C2']
        units_inv = ['Ohm','Ohm','Ohm','s','C2']
        
        return params_inv, names_inv, units_inv
    -----------------
    '''
    def __init__(self):
        return None
    
    def transform_ds(self, Dataset_obj,constants=None):
        if constants is None:
            p, Dataset_obj.params_names_, Dataset_obj.params_units_ = self.transform(Dataset_obj.params_var_)
        else:
            p, Dataset_obj.params_names_, Dataset_obj.params_units_ = self.transform(Dataset_obj.params_var_,constants)
        if isinstance(p,list):
            p = np.array(p)
            np.hstack(p)
            if np.size(p,axis=0) != np.size(Dataset_obj.params_var_,axis=0):
                p = np.transpose(p)
        else:
            raise Exception('Please return a list for the transformed parameters')
        Dataset_obj.params_var_ = p
