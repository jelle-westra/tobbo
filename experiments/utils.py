import matplotlib.pyplot as plt

def set_plt_template() :
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.rc('font', family='serif', serif="cmr10", size=18)
    plt.rc('mathtext', fontset='cm', rm='serif')
    plt.rc('axes', unicode_minus=False)

    plt.rcParams['axes.formatter.use_mathtext'] = True