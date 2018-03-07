#! /bin/bash

# vary population size
jython nn_experiments.py --oa GA --iterations 3000 --ga_p 50 --ga_ma 10 --ga_mu 10
jython nn_experiments.py --oa GA --iterations 3000 --ga_p 100 --ga_ma 10 --ga_mu 10
jython nn_experiments.py --oa GA --iterations 3000 --ga_p 150 --ga_ma 10 --ga_mu 10
jython nn_experiments.py --oa GA --iterations 3000 --ga_p 200 --ga_ma 10 --ga_mu 10

# vary # of mates
jython nn_experiments.py --oa GA --iterations 3000 --ga_p 50 --ga_ma 10 --ga_mu 10
jython nn_experiments.py --oa GA --iterations 3000 --ga_p 50 --ga_ma 20 --ga_mu 10
jython nn_experiments.py --oa GA --iterations 3000 --ga_p 50 --ga_ma 30 --ga_mu 10
jython nn_experiments.py --oa GA --iterations 3000 --ga_p 50 --ga_ma 40 --ga_mu 10

# vary # of mutations
jython nn_experiments.py --oa GA --iterations 3000 --ga_p 50 --ga_ma 10 --ga_mu 10
jython nn_experiments.py --oa GA --iterations 3000 --ga_p 50 --ga_ma 10 --ga_mu 20
jython nn_experiments.py --oa GA --iterations 3000 --ga_p 50 --ga_ma 10 --ga_mu 30
jython nn_experiments.py --oa GA --iterations 3000 --ga_p 50 --ga_ma 10 --ga_mu 40
