#! /bin/bash

# run randomized hill climbing
jython opt_experiments.py --oa RHC

# # run simulated annealing, varying temperature
jython opt_experiments.py --oa SA --sa_cr 0.25

# # run genetic algorithms, vary population and mutation rate
jython opt_experiments.py --oa GA --ga_p 50 --ga_ma 10 --ga_mu 10

# run MIMIC, varying m
jython opt_experiments.py --oa MIMIC --mm 0.3

# aggregate datasets and generate plots
# python opt_plots.py