import histlite as hl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from ctapipe.visualization import CameraDisplay

ANA_BASE = '/home/gerritr/ECAP/nsb_simulation/nsb_refactor'

### Define color palettes for different styles
styles = {'dark': ANA_BASE + '/assets/darkmode.mplstyle',
          'light': ANA_BASE + '/assets/lightmode.mplstyle'}

### Define a style wrapper
def style(func):
    def wrapper(*args, **kwargs):
        if 'style' in kwargs:
            style = styles[kwargs.pop('style')]
        else:
            style = 'seaborn-bright'
        with plt.style.context(style):
            return func(*args, **kwargs)
    return wrapper

### Define a preliminary wrapper
def prelim(pos):
    def decorator(func):
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            if type(res[1]) is np.ndarray:
                for ax in res[1]:
                    ax.add_artist(mpl.offsetbox.AnchoredText("Preliminary",loc=pos,
                                                prop=dict(color='red'), frameon=False))
            else:
                res[1].add_artist(mpl.offsetbox.AnchoredText("Preliminary",loc=pos,
                                                prop=dict(color='red'), frameon=False))
            return res
        return wrapper
    return decorator

def comp_plot_001(pipe, mc, data, title):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), gridspec_kw = {'height_ratios':[2.7,2]})

    # Plot two main plots
    disp1 = CameraDisplay(pipe.config['cam'], ax=ax[0,0])
    disp1.add_colorbar(label='NSB Rate [MHz]', orientation='horizontal')
    nan_index = data[data.isna()].index.tolist()
    disp1.image = data.fillna(np.nan).values

    maxval = np.nanpercentile(np.ma.filled(disp1.image, np.nan), 99.9)

    disp2 = CameraDisplay(pipe.config['cam'], ax=ax[0,1])
    disp2.add_colorbar(label='NSB Rate [MHz]', orientation='horizontal')
    disp2.image = mc

    disp1.set_limits_minmax(0, maxval)
    disp2.set_limits_minmax(0, maxval)

    # Plot histogram/correlation
    h1 = hl.hist(disp1.image, bins=100, range=(0, maxval))
    h2 = hl.hist(disp2.image, bins=h1.bins)
    hl.plot1d(ax[1,0], h1, label='Real Data')
    hl.plot1d(ax[1,0], h2, label='Simulation')

    ax[1,0].legend()
    ax[1,0].set_xlabel('MHz')
    ax[1,0].set_ylabel('Pixel Count')

    ax[1,1].scatter(disp1.image, disp2.image, alpha=0.3)
    ax[1,1].plot([0,maxval],[0,maxval], ls='--', color='grey')
    ax[1,1].set_xlim(0, maxval)
    ax[1,1].set_ylim(0, maxval)
    ax[1,1].set_xlabel('Real Rate [MHz]')
    ax[1,1].set_ylabel('Simulated Rate [MHz]')

    # Super Figure settings
    fig.suptitle(title)
    fig.tight_layout(pad=0.5)
    ax[0,0].set_title('Real')
    ax[0,1].set_title('Simulated')
    
    return fig, ax

def comp_plot_002(cam, mc1, mc2, data, title, labels):
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 10), gridspec_kw = {'height_ratios':[2.7,2]})

    # Plot three main plots
    disp1 = CameraDisplay(cam, ax=ax[0,2])
    disp1.add_colorbar(label='NSB Rate [MHz]', orientation='horizontal')
    nan_index = data[data.isna()].index.tolist()
    disp1.image = data.fillna(np.nan).values

    maxval = np.nanpercentile(np.ma.filled(disp1.image, np.nan), 99.9)

    disp2 = CameraDisplay(cam, ax=ax[0,0])
    disp2.add_colorbar(label='NSB Rate [MHz]', orientation='horizontal')
    disp2.image = mc1
    
    disp3 = CameraDisplay(cam, ax=ax[0,1])
    disp3.add_colorbar(label='NSB Rate [MHz]', orientation='horizontal')
    disp3.image = mc2

    disp1.set_limits_minmax(0, maxval)
    disp2.set_limits_minmax(0, maxval)
    disp3.set_limits_minmax(0, maxval)

    # Plot histogram   
    a = (disp2.image - disp1.image)/disp1.image
    b = (disp3.image - disp1.image)/disp1.image
    h1 = hl.hist(a, bins=51, range=(-1, 1))
    h2 = hl.hist(b, bins=h1.bins)
    hl.plot1d(ax[1,0], h1, label=labels[0] + "\n" + r"$\mu$ = {:.2f}, $\sigma$ = {:.2f}".format(np.mean(a), np.std(a)))
    hl.plot1d(ax[1,0], h2, label=labels[1] + "\n" + r"$\mu$ = {:.2f}, $\sigma$ = {:.2f}".format(np.mean(b), np.std(b)))
    
    ax[1,0].legend()
    ax[1,0].set_xlabel('(Sim-Data)/Data')
    ax[1,0].set_ylabel('Pixel Count')

    # Plot correlation
    ax[1,1].scatter(disp1.image, disp2.image-disp1.image, alpha=0.2)
    ax[1,1].scatter(disp1.image, disp3.image-disp1.image, alpha=0.2)
    ax[1,1].plot([0,maxval],[0, 0], ls='--', color='grey')
    
    ax[1,1].set_xlim(0, maxval)
    ax[1,1].set_ylim(-300, 300)
    ax[1,1].set_xlabel('Real Rate [MHz]')
    ax[1,1].set_ylabel('(Sim - Data) [MHz]')
    
    # Plot change
    hdiff = hl.hist((mc2-mc1)/(data-mc1), bins=51, range=(-3,3))
    hl.plot1d(ax[1,2], hdiff, label='Relative Change')
    
    ax[1,2].set_xlabel('(Sim2-Sim1)/(Data-Sim1)')
    ax[1,2].set_ylabel('Pixel Counts')
    
    # Super Figure settings
    fig.suptitle(title, y=1.02, size=14)
    fig.tight_layout(pad=0.5)
    ax[0,0].set_title(labels[0])
    ax[0,1].set_title(labels[1])
    ax[0,2].set_title(labels[2])
    
    return fig, ax

def comp_plot_003(pipe, mc, data, title):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 12))

    # Plot two main plots
    disp1 = CameraDisplay(pipe.config['cam'], ax=ax[0,0])
    disp1.add_colorbar(label='Data NSB Rate [MHz]', orientation='horizontal')
    nan_index = data[data.isna()].index.tolist()
    disp1.image = data.fillna(np.nan).values

    maxval = np.nanpercentile(np.ma.filled(disp1.image, np.nan), 99.5)
    
    disp2 = CameraDisplay(pipe.config['cam'], ax=ax[1,0])
    disp2.add_colorbar(label='MC NSB Rate [MHz]', orientation='horizontal')
    disp2.image = mc

    disp1.set_limits_minmax(0, maxval)
    disp2.set_limits_minmax(0, maxval)
    
    cmap = mpl.cm.coolwarm
    bounds = [-1, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
    
    disp3 = CameraDisplay(pipe.config['cam'], ax=ax[0,1])
    disp3.add_colorbar(label='(Data-MC)/Data', orientation='horizontal')
    disp3.norm = norm
    disp3.image = 1 - disp2.image/disp1.image
    
    im = ax[1,1].scatter(disp1.image, disp2.image, alpha=1, c=1- disp2.image/disp1.image, norm=norm)
    ax[1,1].plot([0,maxval],[0,maxval], ls='--', color='grey')
    ax[1,1].set_xlim(0, maxval)
    ax[1,1].set_ylim(0, maxval)
    ax[1,1].set_xlabel('Real Rate [MHz]')
    ax[1,1].set_ylabel('Simulated Rate [MHz]')
    
    fig.colorbar(im, ax=ax[1,1], orientation='horizontal', label='(Data-MC)/Data')

    # Super Figure settings
    fig.suptitle(title)
    fig.tight_layout(pad=0.5)
    
    return fig, ax