import matplotlib.pyplot as plt


def set_style(plt):
    plt.style.use('seaborn')
    plt.rcParams['legend.numpoints'] = 1
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['legend.fontsize'] = 10
    return plt


legend_opts = {
    'fontsize': 15,
    'frameon': True,
    'framealpha': 1,
    'facecolor': 'white',
    'edgecolor': 'black',
    'labelspacing': 0.1}


def savefig(fig, filename, **kwargs):
    fig.savefig(f'../viz/{filename}', bbox_inches='tight', **kwargs)


plt = set_style(plt)
