"""
Code to generate plots related to generic optimization problem experiments.

"""
from helpers import get_abspath
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context('notebook', rc={'lines.markeredgewidth': 1.0})
sns.set_style("darkgrid")


def group_results(resfile, aggtype='mean'):
    """Groups experiment results by iteration using the specified aggregation
    function and replaces the old results file.

    Args:
        resfile (str): Path to results file.
        aggtype (str): Aggregation function.

    """
    df = pd.read_csv(resfile)
    grouped = None
    if aggtype == 'mean':
        grouped = df.groupby('iterations').mean()
    elif aggtype == 'max':
        grouped = df.groupby('iterations').max()

    # remove old file
    try:
        os.remove(resfile)
    except Exception as e:
        print e
        pass

    # save grouped results
    grouped.to_csv(path_or_buf=resfile)


def plot_curves(rhcRes, saRes, gaRes, mimicRes, opName, c):
    """Plots fitness function values vs. number of iterations for all four
    optimization algorithms

    Args:
        rhcRes (str): Filepath for randomized hill climbing results.
        saRes (str): Filepath for simulated annealing results.
        gaRes (str): Filepath for genetic algorithms results.
        mimicRes (str): Filepath for MIMIC results.
        c (str): Type of curve (fitness, timing, or eval function calls).
        opName (str): Optimization problem name/filepath.

    """
    opTitles = {'TSP': 'Travelling Salesman Problem',
                'FF': 'Flip Flop', 'CP': 'Continuous Peaks'}
    cTypes = {'fitness': 'Fitness Value',
              'time': 'Running Time', 'fevals': 'Eval Function Calls'}
    yLabels = {'fitness': 'Fitness Value',
               'time': 'Log Running Time', 'fevals': 'Log Eval Func Calls'}
    opTitle = opTitles[opName]
    cTitle = cTypes[c]
    yLabel = yLabels[c]

    # get results
    rhc = pd.read_csv(get_abspath(rhcRes, 'results/OPT/{}'.format(opName)))
    sa = pd.read_csv(get_abspath(saRes, 'results/OPT/{}'.format(opName)))
    ga = pd.read_csv(get_abspath(gaRes, 'results/OPT/{}'.format(opName)))
    mm = pd.read_csv(get_abspath(mimicRes, 'results/OPT/{}'.format(opName)))

    iters = rhc['iterations']
    fRHC = None
    fSA = None
    fGA = None
    fMIMIC = None
    if c in ('fevals, time'):
        fRHC = np.log(rhc[c])
        fSA = np.log(sa[c])
        fGA = np.log(ga[c])
        fMIMIC = np.log(mm[c])
    else:
        fRHC = rhc[c]
        fSA = sa[c]
        fGA = ga[c]
        fMIMIC = mm[c]

    # create fitness/timing/fevals curve
    plt.figure(0)
    plt.plot(iters, fRHC, color='k', label='RHC')
    plt.plot(iters, fSA, color='r', label='SA')
    plt.plot(iters, fGA, color='g', label='GA')
    plt.plot(iters, fMIMIC, color='b', label='MIMIC')
    plt.xlim(xmin=0)
    plt.legend(loc='best')
    plt.grid(color='grey', linestyle='dotted')
    plt.xlabel('Training Iterations')
    plt.title('{} - {} vs. Iterations'.format(opTitle, cTitle))
    plt.ylabel(yLabel)

    # save learning curve plot as PNG
    plotdir = 'plots/OPT/{}'.format(opName)
    plotpath = get_abspath('{}_curve.png'.format(c), plotdir)
    plt.savefig(plotpath, bbox_inches='tight')
    plt.clf()


def problem_complexity(opName, p='N'):
    """Creates problem complexity/size curves for the specified optimization
    problem.

    Args:
        opName (str): Name of optimization problem.
        p (str): Name of problem parameter.

    """
    opTitles = {'TSP': 'Travelling Salesman Problem',
                'FF': 'Flip Flop', 'CP': 'Continuous Peaks'}
    resdir = 'results/OPT/ProblemComplexity'
    pTitle = opTitles[opName]

    df = pd.read_csv(get_abspath('{}.csv'.format(opName), resdir))
    X = df[p]
    RHC = df['RHC']
    SA = df['SA']
    GA = df['GA']
    MIMIC = df['MIMIC']

    # create figure
    plt.figure(0)
    plt.plot(X, RHC, color='k', label='RHC')
    plt.plot(X, SA, color='r', label='SA')
    plt.plot(X, GA, color='g', label='GA')
    plt.plot(X, MIMIC, color='b', label='MIMIC')
    plt.legend(loc='best')
    plt.grid(color='grey', linestyle='dotted')
    plt.xlabel(p)
    plt.xticks(X)
    plt.title('{} - Fitness Score vs. Problem Size'.format(pTitle))
    plt.ylabel('Fitness Score')

    # save problem complexity plot as PNG
    plotdir = 'plots/OPT/ProblemComplexity'
    plotpath = get_abspath('{}_pcCurve.png'.format(opName), plotdir)
    plt.savefig(plotpath, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    # group all results datasets
    p = os.path.abspath(os.path.join(os.curdir, os.pardir, 'results/OPT/Archive'))
    for root, dirs, files in os.walk(p):
        for file in files:
            if file.endswith('results.csv'):
                group_results(resfile=os.path.join(root, file), aggtype='max')

    # get result file names
    rhcRes = 'RHC_results.csv'
    saRes = 'SA_0.25_results.csv'
    gaRes = 'GA_50_10_10_results.csv'
    mimicRes = 'MIMIC_100_50_0.3_results.csv'

    # plot fitness curve
    plot_curves(rhcRes, saRes, gaRes, mimicRes, opName='TSP', c='fitness')
    plot_curves(rhcRes, saRes, gaRes, mimicRes, opName='FF', c='fitness')
    plot_curves(rhcRes, saRes, gaRes, mimicRes, opName='CP', c='fitness')

    plot_curves(rhcRes, saRes, gaRes, mimicRes, opName='TSP', c='time')
    plot_curves(rhcRes, saRes, gaRes, mimicRes, opName='FF', c='time')
    plot_curves(rhcRes, saRes, gaRes, mimicRes, opName='CP', c='time')

    plot_curves(rhcRes, saRes, gaRes, mimicRes, opName='TSP', c='fevals')
    plot_curves(rhcRes, saRes, gaRes, mimicRes, opName='FF', c='fevals')
    plot_curves(rhcRes, saRes, gaRes, mimicRes, opName='CP', c='fevals')

    # plot problem complexity
    problem_complexity(opName='TSP', p='N')
    problem_complexity(opName='CP', p='T')
    problem_complexity(opName='FF', p='N')
