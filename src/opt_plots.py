"""
Code to generate plots related to generic optimization problem experiments.

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

