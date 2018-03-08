#! /bin/bash

# run randomized hill climbing
jython opt_experiments.py --oa RHC

# run simulated annealing, varying temperature
jython opt_experiments.py --oa SA --sa_cr 0.15
jython opt_experiments.py --oa SA --sa_cr 0.30
jython opt_experiments.py --oa SA --sa_cr 0.45
jython opt_experiments.py --oa SA --sa_cr 0.60
jython opt_experiments.py --oa SA --sa_cr 0.75

# run genetic algorithms, vary population and mutation rate
jython opt_experiments.py --oa GA --ga_p 50
jython opt_experiments.py --oa GA --ga_p 75
jython opt_experiments.py --oa GA --ga_p 100

jython opt_experiments.py --oa GA --ga_mu 10
jython opt_experiments.py --oa GA --ga_mu 20
jython opt_experiments.py --oa GA --ga_mu 30

# run MIMIC, varying m
jython opt_experiments.py --oa MIMIC --mm 0.1
jython opt_experiments.py --oa MIMIC --mm 0.3
jython opt_experiments.py --oa MIMIC --mm 0.5
jython opt_experiments.py --oa MIMIC --mm 0.7

# aggregate datasets and generate plots
python opt_plots.py