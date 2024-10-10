import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def plot_results(titles, axes_units, y_test, y_pred, model_name, data_name,savefig=False,savepath=''):
    n_plots = len(titles)
    per_row = np.ceil(np.sqrt(n_plots)).astype(int)
    if n_plots <= per_row*(per_row-1):
        #print('Reducing Plot Size')
        fig,ax = plt.subplots(per_row - 1,per_row,figsize=(10,10))
    else:
        fig,ax = plt.subplots(per_row,per_row,figsize=(10,10))
    
    nrows = 0
    ncols = 0
    
    for i in range(n_plots):
        ax[nrows][ncols].scatter(y_test[:,i],y_pred[:,i],s=.1,c='black')
        ax[nrows][ncols].set_title(titles[i])
        r2 = metrics.r2_score(y_test[:,i],y_pred[:,i])
        rmse = metrics.root_mean_squared_error(y_test[:,i],y_pred[:,i])
        ax[nrows][ncols].text(0.05, 0.95, '$r^2$: {:.3f}\n$RMSE$: {:.2e}'.format(r2, rmse), horizontalalignment='left',
         verticalalignment='top', transform=ax[nrows][ncols].transAxes)
        ax[nrows][ncols].ticklabel_format(axis='both',style='sci',scilimits=(-1,2))
        ax[nrows][ncols].set_xlim(0.9*np.min(y_test[:,i]),1.02*np.max(y_test[:,i]))
        ax[nrows][ncols].set_ylim(0.9*np.min(y_test[:,i]),1.02*np.max(y_test[:,i]))
        ax[nrows][ncols].set_xlabel('Actual ('+axes_units[i]+')')
        ax[nrows][ncols].set_ylabel('Predicted ('+axes_units[i]+')')
        if ncols != per_row - 1:
            ncols += 1
        else:
            ncols = 0
            nrows += 1
    fig.suptitle(model_name + ': ' + data_name)
    plt.tight_layout()
    plt.savefig(model_name + '_' + data_name,dpi=1200)
    plt.show()
    
def plot_model(ModelDS,savefig=False,savepath='',**kwargs):
    titles = kwargs.get('titles',ModelDS.params_names_)
    axes_units = kwargs.get('axes_units',ModelDS.params_units_)
    model_name = kwargs.get('model_name',ModelDS.name_)
    data_name = kwargs.get('data_name',ModelDS.dataname_)
    
    plot_results(titles,axes_units,ModelDS.p_test_,ModelDS.p_test_pred_,model_name,data_name,savefig=savefig,savepath=savepath)