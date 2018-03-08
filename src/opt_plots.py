"""
Code to generate plots related to generic optimization problem experiments.

"""
from helpers import get_abspath
import os
import pandas as pd
import numpy as np
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


if __name__ == '__main__':
    # group all results datasets
    p = os.path.abspath(os.path.join(os.curdir, os.pardir, 'results/OPT'))
    for root, dirs, files in os.walk(p):
        for file in files:
            if file.endswith('results.csv'):
                group_results(resfile=os.path.join(root, file))
