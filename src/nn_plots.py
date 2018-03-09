"""
Code to generate plots related to neural network weight optimization
experiments.

"""
from helpers import get_abspath, save_dataset
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context('notebook', rc={'lines.markeredgewidth': 1.0})
sns.set_style("darkgrid")


def combine_datasets(df):
    """Creates a combined dataset for error and accuracy to compare various
    optimization algorithms and saves it as a CSV file.

    Args:
        dfs (dict(Pandas.DataFrame)): Dictionary of data frames.

    """
    # create combined error datasets
    BP = df['BP']
    RHC = datafiles['RHC']
    SA = datafiles['SA']
    GA = datafiles['GA']

    # rename columns
    bCols = {'MSE_train': 'bp_msetrain', 'MSE_test': 'bp_msetest', 'MSE_validation': 'bp_msevalid', 'acc_train': 'bp_acctrain', 'acc_test': 'bp_acctest', 'acc_validation': 'bp_accvalid', 'seconds_elapsed': 'bp_time'}
    rCols = {'MSE_train': 'rhc_msetrain', 'MSE_test': 'rhc_msetest', 'MSE_validation': 'rhc_msevalid', 'acc_train': 'rhc_acctrain', 'acc_test': 'rhc_acctest', 'acc_validation': 'rhc_accvalid', 'seconds_elapsed': 'rhc_time'}
    sCols = {'MSE_train': 'sa_msetrain', 'MSE_test': 'sa_msetest', 'MSE_validation': 'sa_msevalid', 'acc_train': 'sa_acctrain', 'acc_test': 'sa_acctest', 'acc_validation': 'sa_accvalid', 'seconds_elapsed': 'sa_time'}
    gCols = {'MSE_train': 'ga_msetrain', 'MSE_test': 'ga_msetest', 'MSE_validation': 'ga_msevalid', 'acc_train': 'ga_acctrain', 'acc_test': 'ga_acctest', 'acc_validation': 'ga_accvalid', 'seconds_elapsed': 'ga_time'}

    BP = df['BP'].rename(index=str, columns=bCols)
    RHC = df['RHC'].drop(columns='iteration').rename(index=str, columns=rCols)
    SA = df['SA'].drop(columns='iteration').rename(index=str, columns=sCols)
    GA = df['GA'].drop(columns='iteration').rename(index=str, columns=gCols)

    # create combined datasets
    res = pd.concat([BP, RHC, SA, GA], axis=1)
    save_dataset(res, filename='combined.csv', subdir='results/NN/combined')


def combined_error(df, ef='Mean squared error'):
    """Generates plots for comparing error across the various
    optimization algorithms and saves them as PNG files.""

    Args:
        df (Pandas.DataFrame): Combined results dataset.
        ef (str): Name of loss function.

    """
    # get columns
    iters = df['iteration']
    bp_msetrain = df['bp_msetrain']
    bp_msetest = df['bp_msetest']
    rhc_msetrain = df['rhc_msetrain']
    rhc_msetest = df['rhc_msetest']
    sa_msetrain = df['sa_msetrain']
    sa_msetest = df['sa_msetest']
    ga_msetrain = df['ga_msetrain']
    ga_msetest = df['ga_msetest']

    # create error curve for train dataset
    plt.figure(0)
    plt.plot(iters, bp_msetrain, marker='o',
             markevery=30, color='b', label='Backprop')
    plt.plot(iters, rhc_msetrain, marker='s',
             markevery=30, color='r', label='RHC')
    plt.plot(iters, sa_msetrain, marker='^',
             markevery=30, color='g', label='SA')
    plt.plot(iters, ga_msetrain, marker='v',
             markevery=30, color='k', label='GA')
    plt.xlim(xmin=-30)
    plt.ylim(ymin=-0.025)
    plt.legend(loc='best')
    plt.grid(color='grey', linestyle='dotted')
    plt.title('Algorithm Comparison (training data) - {}'.format(ef))
    plt.xlabel('Iterations')
    plt.ylabel(ef)

    # save learning curve plot as PNG
    plotdir = 'plots/NN/combined'
    plotpath = get_abspath('error_train.png', plotdir)
    plt.savefig(plotpath, bbox_inches='tight')
    plt.clf()

    # create error curve for train dataset
    plt.plot(iters, bp_msetest, marker='o',
             markevery=30, color='b', label='Backprop')
    plt.plot(iters, rhc_msetest, marker='s',
             markevery=30, color='r', label='RHC')
    plt.plot(iters, sa_msetest, marker='^',
             markevery=30, color='g', label='SA')
    plt.plot(iters, ga_msetest, marker='v',
             markevery=30, color='k', label='GA')
    plt.xlim(xmin=-30)
    plt.ylim(ymin=-0.025)
    plt.legend(loc='best')
    plt.grid(color='grey', linestyle='dotted')
    plt.title('Algorithm Comparison (test data) - {}'.format(ef))
    plt.xlabel('Iterations')
    plt.ylabel(ef)

    # save learning curve plot as PNG
    plotdir = 'plots/NN/combined'
    plotpath = get_abspath('error_test.png', plotdir)
    plt.savefig(plotpath, bbox_inches='tight')
    plt.clf()


def combined_acc(df):
    """Generates plots for comparing accuracy scores across the various
    optimization algorithms and saves them as PNG files.

    Args:
        df (Pandas.DataFrame): Combined results dataset.

    """
    # get columns
    iters = df['iteration']
    bp_acctrain = df['bp_acctrain']
    bp_acctest = df['bp_acctest']
    rhc_acctrain = df['rhc_acctrain']
    rhc_acctest = df['rhc_acctest']
    sa_acctrain = df['sa_acctrain']
    sa_acctest = df['sa_acctest']
    ga_acctrain = df['ga_acctrain']
    ga_acctest = df['ga_acctest']

    # create learning curve for train dataset
    plt.figure(0)
    plt.plot(iters, bp_acctrain, marker='o',
             markevery=30, color='b', label='Backprop')
    plt.plot(iters, rhc_acctrain, marker='s',
             markevery=30, color='r', label='RHC')
    plt.plot(iters, sa_acctrain, marker='^',
             markevery=30, color='g', label='SA')
    plt.plot(iters, ga_acctrain, marker='v',
             markevery=30, color='k', label='GA')
    plt.xlim(xmin=-30)
    plt.ylim(ymax=1.025)
    plt.legend(loc='best')
    plt.grid(color='grey', linestyle='dotted')
    plt.title('Algorithm Comparison - Learning Curve (train)')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')

    # save learning curve plot as PNG
    plotdir = 'plots/NN/combined'
    plotpath = get_abspath('acc_train.png', plotdir)
    plt.savefig(plotpath, bbox_inches='tight')
    plt.clf()

    # create learning curve for train dataset
    plt.plot(iters, bp_acctest, marker='o',
             markevery=30, color='b', label='Backprop')
    plt.plot(iters, rhc_acctest, marker='s',
             markevery=30, color='r', label='RHC')
    plt.plot(iters, sa_acctest, marker='^',
             markevery=30, color='g', label='SA')
    plt.plot(iters, ga_acctest, marker='v',
             markevery=30, color='k', label='GA')
    plt.xlim(xmin=-30)
    plt.ylim(ymax=1.025)
    plt.legend(loc='best')
    plt.grid(color='grey', linestyle='dotted')
    plt.title('Algorithm Comparison - Learning Curve (test)')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')

    # save learning curve plot as PNG
    plotdir = 'plots/NN/combined'
    plotpath = get_abspath('acc_test.png', plotdir)
    plt.savefig(plotpath, bbox_inches='tight')
    plt.clf()


def combined_time(df):
    """Generates a plot for comparing elapsed time across the various
    optimization algorithms and saves it as a PNG file.

    Args:
        df (Pandas.DataFrame): Combined results dataset.

    """
    # get columns
    iters = df['iteration']
    bp_time = np.log(df['bp_time'])
    rhc_time = np.log(df['rhc_time'])
    sa_time = np.log(df['sa_time'])
    ga_time = np.log(df['ga_time'])

    # create timing curve for train dataset
    plt.figure(0)
    plt.plot(iters, bp_time, marker='o',
             markevery=30, color='b', label='Backprop')
    plt.plot(iters, rhc_time, marker='s',
             markevery=30, color='r', label='RHC')
    plt.plot(iters, sa_time, marker='^',
             markevery=30, color='g', label='SA')
    plt.plot(iters, ga_time, marker='v',
             markevery=30, color='k', label='GA')
    plt.xlim(xmin=-30)
    plt.legend(loc='best')
    plt.grid(color='grey', linestyle='dotted')
    plt.title('Algorithm Comparison - Elapsed Time')
    plt.xlabel('Iterations')
    plt.ylabel('Log time')

    # save timing curve plot as PNG
    plotdir = 'plots/NN/combined'
    plotpath = get_abspath('timing_curve.png', plotdir)
    plt.savefig(plotpath, bbox_inches='tight')
    plt.clf()


def error_curve(df, oaName, title):
    """Plots the error curve for a given optimization algorithm on the
    seismic-bumps dataset and saves it as a PNG file.

    Args:
        df (Pandas.DataFrame): Dataset.
        oaName (str): Name of optimization algorithm.
        title (str): Plot title.
        ef (str): Name of loss function.

    """

    # get columns
    iterations = df['iteration']
    MSE_train = df['MSE_train']
    MSE_test = df['MSE_test']
    MSE_valid = df['MSE_validation']

    # create error curve
    plt.figure(0)
    plt.plot(iterations, MSE_train, color='b', label='Training')
    plt.plot(iterations, MSE_test, color='r', label='Test')
    plt.plot(iterations, MSE_valid, color='g', label='Validation')
    plt.xlim(xmin=-30)
    plt.legend(loc='best')
    plt.grid(color='grey', linestyle='dotted')
    plt.title('{} - Mean squared error'.format(title))
    plt.xlabel('Iterations')
    plt.ylabel('Mean squared error')

    # save learning curve plot as PNG
    plotdir = 'plots/NN'
    plot_tgt = '{}/{}'.format(plotdir, oaName)
    plotpath = get_abspath('{}_error.png'.format(oaName), plot_tgt)
    plt.savefig(plotpath, bbox_inches='tight')
    plt.clf()


def learning_curve(df, oaName, title):
    """Plots the learning curve for a given optimization algorithm on the
    seismic-bumps dataset and saves it as a PNG file.

    Args:
        df (Pandas.DataFrame): Dataset.
        oaName (str): Name of optimization algorithm.
        title (str): Plot title.

    """

    # get columns
    iterations = df['iteration']
    acc_train = df['acc_train']
    acc_test = df['acc_test']
    acc_valid = df['acc_validation']

    # create learning curve
    plt.figure(0)
    plt.plot(iterations, acc_train, color='b', label='Training')
    plt.plot(iterations, acc_test, color='r', label='Test')
    plt.plot(iterations, acc_valid, color='g', label='Validation')
    plt.xlim(xmin=-30)
    plt.legend(loc='best')
    plt.grid(color='grey', linestyle='dotted')
    plt.title('{} - Learning Curve'.format(title))
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')

    # save learning curve plot as PNG
    plotdir = 'plots/NN'
    plot_tgt = '{}/{}'.format(plotdir, oaName)
    plotpath = get_abspath('{}_LC.png'.format(oaName), plot_tgt)
    plt.savefig(plotpath, bbox_inches='tight')
    plt.clf()


def sa_validation_curve():
    """Plots the cooling rate validation curve for the simulated annealing
    algorithm and saves it as a PNG file.

    """
    # load datasets
    resdir = 'results/NN/SA'
    df_15 = pd.read_csv(get_abspath('results_10000000000.0_0.15.csv', resdir))
    df_30 = pd.read_csv(get_abspath('results_10000000000.0_0.3.csv', resdir))
    df_45 = pd.read_csv(get_abspath('results_10000000000.0_0.45.csv', resdir))
    df_60 = pd.read_csv(get_abspath('results_10000000000.0_0.6.csv', resdir))
    df_75 = pd.read_csv(get_abspath('results_10000000000.0_0.75.csv', resdir))
    df_90 = pd.read_csv(get_abspath('results_10000000000.0_0.9.csv', resdir))

    # get columns
    iters = df_15['iteration']
    train_15 = df_15['MSE_train']
    test_15 = df_15['MSE_test']
    train_30 = df_30['MSE_train']
    test_30 = df_30['MSE_test']
    train_45 = df_45['MSE_train']
    test_45 = df_45['MSE_test']
    train_60 = df_60['MSE_train']
    test_60 = df_60['MSE_test']
    train_75 = df_75['MSE_train']
    test_75 = df_75['MSE_test']
    train_90 = df_90['MSE_train']
    test_90 = df_90['MSE_test']

    # create validation curve for training data
    plt.figure(0)
    plt.plot(iters, train_15, color='b', label='CR: 0.15')
    plt.plot(iters, train_30, color='g', label='CR: 0.30')
    plt.plot(iters, train_45, color='r', label='CR: 0.45')
    plt.plot(iters, train_60, color='c', label='CR: 0.60')
    plt.plot(iters, train_75, color='k', label='CR: 0.75')
    plt.plot(iters, train_90, color='m', label='CR: 0.90')
    plt.legend(loc='best')
    plt.grid(color='grey', linestyle='dotted')
    plt.title('SA Validation Curve - Cooling Rate (train)')
    plt.xlabel('Iterations')
    plt.ylabel('Mean squared error')

    # save train validation curve plot as PNG
    plotdir = 'plots/NN/SA'
    plotpath = get_abspath('SA_CR_train.png', plotdir)
    plt.savefig(plotpath, bbox_inches='tight')
    plt.clf()

    # create validation curve for test data
    plt.figure(0)
    plt.plot(iters, test_15, color='b', label='CR: 0.15')
    plt.plot(iters, test_30, color='g', label='CR: 0.30')
    plt.plot(iters, test_45, color='r', label='CR: 0.45')
    plt.plot(iters, test_60, color='c', label='CR: 0.60')
    plt.plot(iters, test_75, color='k', label='CR: 0.75')
    plt.plot(iters, test_90, color='m', label='CR: 0.90')
    plt.legend(loc='best')
    plt.grid(color='grey', linestyle='dotted')
    plt.title('SA Validation Curve - Cooling Rate (test)')
    plt.xlabel('Iterations')
    plt.ylabel('Mean squared error')

    # save test validation curve plot as PNG
    plotdir = 'plots/NN/SA'
    plotpath = get_abspath('SA_CR_test.png', plotdir)
    plt.savefig(plotpath, bbox_inches='tight')
    plt.clf()


def ga_population_curve():
    """Plots the population size validation curve for genetic algorithms and
    saves it as a PNG file.

    """
    # load datasets
    resdir = 'results/NN/GA'
    df_50 = pd.read_csv(get_abspath('results_50_10_10.csv', resdir))
    df_100 = pd.read_csv(get_abspath('results_100_10_10.csv', resdir))
    df_150 = pd.read_csv(get_abspath('results_150_10_10.csv', resdir))
    df_200 = pd.read_csv(get_abspath('results_200_10_10.csv', resdir))

    # get columns
    iters = df_50['iteration']
    train_50 = df_50['MSE_train']
    test_50 = df_50['MSE_test']
    train_100 = df_100['MSE_train']
    test_100 = df_100['MSE_test']
    train_150 = df_150['MSE_train']
    test_150 = df_150['MSE_test']
    train_200 = df_200['MSE_train']
    test_200 = df_200['MSE_test']

    # create validation curve for training data
    plt.figure(0)
    plt.plot(iters, train_50, color='b', label='Population: 50')
    plt.plot(iters, train_100, color='g', label='Population: 100')
    plt.plot(iters, train_150, color='r', label='Population: 150')
    plt.plot(iters, train_200, color='r', label='Population: 200')
    plt.xlim(xmin=-30)
    plt.legend(loc='best')
    plt.grid(color='grey', linestyle='dotted')
    plt.title('GA Validation Curve - Population Size (train)')
    plt.xlabel('Iterations')
    plt.ylabel('Mean squared error')

    # save complexity curve plot as PNG
    plotdir = 'plots/NN/GA'
    plotpath = get_abspath('GA_P_train.png', plotdir)
    plt.savefig(plotpath, bbox_inches='tight')
    plt.clf()

    # create complexity curve for test data
    plt.figure(0)
    plt.plot(iters, test_50, color='b', label='Population: 50')
    plt.plot(iters, test_100, color='g', label='Population: 100')
    plt.plot(iters, test_150, color='r', label='Population: 150')
    plt.plot(iters, test_200, color='r', label='Population: 200')
    plt.xlim(xmin=-30)
    plt.legend(loc='best')
    plt.grid(color='grey', linestyle='dotted')
    plt.title('GA Validation Curve - Population Size (test)')
    plt.xlabel('Iterations')
    plt.ylabel('Mean squared error')

    # save learning curve plot as PNG
    plotdir = 'plots/NN/GA'
    plotpath = get_abspath('GA_P_test.png', plotdir)
    plt.savefig(plotpath, bbox_inches='tight')
    plt.clf()


def ga_mating_curve():
    """Plots the mating rate validation curve for genetic algorithms and
    saves it as a PNG file.

    """
    # load datasets
    resdir = 'results/NN/GA'
    df_10= pd.read_csv(get_abspath('results_50_10_10.csv', resdir))
    df_20 = pd.read_csv(get_abspath('results_50_20_10.csv', resdir))
    df_30 = pd.read_csv(get_abspath('results_50_30_10.csv', resdir))

    # get columns
    iters = df_10['iteration']
    train_10 = df_10['MSE_train']
    test_10 = df_10['MSE_test']
    train_20 = df_20['MSE_train']
    test_20 = df_20['MSE_test']
    train_30 = df_30['MSE_train']
    test_30 = df_30['MSE_test']

    # create validation curve for training data
    plt.figure(0)
    plt.plot(iters, train_10, color='b', label='# of mates: 10')
    plt.plot(iters, train_20, color='g', label='# of mates: 20')
    plt.plot(iters, train_30, color='r', label='# of mates: 30')
    plt.xlim(xmin=-30)
    plt.legend(loc='best')
    plt.grid(color='grey', linestyle='dotted')
    plt.title('GA Validation Curve - Mating Rate (train)')
    plt.xlabel('Iterations')
    plt.ylabel('Mean squared error')

    # save complexity curve plot as PNG
    plotdir = 'plots/NN/GA'
    plotpath = get_abspath('GA_MA_train.png', plotdir)
    plt.savefig(plotpath, bbox_inches='tight')
    plt.clf()

    # create complexity curve for test data
    plt.figure(0)
    plt.plot(iters, test_10, color='b', label='# of mates: 10')
    plt.plot(iters, test_20, color='g', label='# of mates: 20')
    plt.plot(iters, test_30, color='r', label='# of mates: 30')
    plt.xlim(xmin=-30)
    plt.legend(loc='best')
    plt.grid(color='grey', linestyle='dotted')
    plt.title('GA Validation Curve - Mating Rate (test)')
    plt.xlabel('Iterations')
    plt.ylabel('Mean squared error')

    # save learning curve plot as PNG
    plotdir = 'plots/NN/GA'
    plotpath = get_abspath('GA_MA_test.png', plotdir)
    plt.savefig(plotpath, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    # get datasets for error and accuracy curves
    resdir = 'results/NN'
    BP = pd.read_csv(get_abspath('BP/results.csv', resdir))
    RHC = pd.read_csv(get_abspath('RHC/results.csv', resdir))
    SA = pd.read_csv(get_abspath('SA/results_10000000000.0_0.3.csv', resdir))
    GA = pd.read_csv(get_abspath('GA/results_100_10_10.csv', resdir))

    # generate error curves
    error_curve(BP, oaName='BP', title='Backpropagation')
    error_curve(RHC, oaName='RHC', title='Randomized Hill Climbing')
    error_curve(SA, oaName='SA', title='Simulated Annealing')
    error_curve(GA, oaName='GA', title='Genetic Algorithms')

    # generate validation curves
    learning_curve(BP, oaName='BP', title='Backpropagation')
    learning_curve(RHC, oaName='RHC', title='Randomized Hill Climbing')
    learning_curve(SA, oaName='SA', title='Simulated Annealing')
    learning_curve(GA, oaName='GA', title='Genetic Algorithms')

    # generate validation curves
    sa_validation_curve()
    ga_population_curve()
    ga_mating_curve()

    # generate combined dataset
    datafiles = {'BP': BP, 'RHC': RHC, 'SA': SA, 'GA': GA}
    combine_datasets(datafiles)

    # generated combined plots
    combined = pd.read_csv(get_abspath('combined/combined.csv', resdir))
    combined_error(combined)
    combined_acc(combined)
    combined_time(combined)
