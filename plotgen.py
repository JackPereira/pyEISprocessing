import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def plot_results(titles, axes_units, y_test, y_pred, model_name, data_name,
                 logswitch = 2, savefig=False,savepath=''):
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
        if logswitch is not None:
            if np.log10(np.max(y_test[:,i])) - np.log10(np.min(y_test[:,i])) > logswitch:
                ax[nrows][ncols].set_yscale('log')
                ax[nrows][ncols].set_xscale('log')
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
    
def plot_comparison(losses, scores, suptitle=None, score_name=None, loss_name=None,labels=None, 
                    plot_std=True,savefig=False,savepath='',dpi=800):
    
    ncomp = len(losses)
    cmap = plt.get_cmap('turbo')
    colors = [cmap(x/(ncomp-1)) for x in range(ncomp)]
    if labels is None:
        labels = np.arange(1,ncomp+1,1).astype(str)
    t = np.arange(1,np.size(losses[0],axis=1)+1,1)
    plt.close('all')
    if scores is None:
        maxlmean = 0
        fig,ax = plt.subplots(figsize=(10,10))
        for i,loss in enumerate(losses):
            lmean = np.mean(loss,axis=0)
            if np.max(lmean) > maxlmean:
                maxlmean = np.max(lmean)
            ax.scatter(t,lmean,s=3,color=colors[i],label=labels[i])
            if plot_std:
                lstd = np.std(loss,axis=0)
                ax.fill_between(t, lmean - lstd, lmean + lstd, interpolate=True,alpha=0.2,color=colors[i])
        ax.set_ylabel('Loss Function')
        ax.set_xlabel('Epoch')
        ax.set_ylim((0,1.1*maxlmean))
        ax.legend()
    else:
        maxlmean = 0
        maxsmean = 0
        minsmean = 0
        fig,ax = plt.subplots(1,2,figsize=(15,7))
        for i,(loss,score) in enumerate(zip(losses,scores)):
            lmean, smean = np.mean(loss,axis=0),np.mean(score,axis=0)
            if np.max(lmean) > maxlmean:
                maxlmean = np.max(lmean)
            if np.max(smean) > maxsmean:
                maxsmean = np.max(smean)
            if np.min(smean) < minsmean:
                minsmean = np.min(smean)
            ax[0].scatter(t,lmean,s=3,label=labels[i],color=colors[i])
            ax[1].scatter(t,smean,s=3,label=labels[i],color=colors[i])
            if plot_std:
                lstd, sstd = np.std(loss,axis=0),np.std(score,axis=0)
                ax[0].fill_between(t, lmean - lstd, lmean + lstd, interpolate=True,alpha=0.2,color=colors[i])
                ax[1].fill_between(t, smean - sstd, smean + sstd, interpolate=True,alpha=0.2,color=colors[i])
        ax[0].set_ylabel('Loss Function')
        ax[1].set_ylabel('Score Function')
        ax[0].set_xlabel('Epoch')
        ax[1].set_xlabel('Epoch')
        ax[0].set_ylim((0,1.1*maxlmean))
        ax[1].set_ylim((0.9*minsmean,1.1*maxsmean))
        ax[0].legend(loc='upper right')
        ax[1].legend()
    if suptitle is not None:
        plt.suptitle(suptitle)
    if savefig:
        plt.savefig(savepath,dpi=dpi)
    plt.show()
