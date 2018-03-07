#! /bin/bash

# vary cooling rate
jython nn_experiments.py --oa SA --iterations 3000 --sa_t 10000000000.0 --sa_c 0.15
jython nn_experiments.py --oa SA --iterations 3000 --sa_t 10000000000.0 --sa_c 0.30
jython nn_experiments.py --oa SA --iterations 3000 --sa_t 10000000000.0 --sa_c 0.45
jython nn_experiments.py --oa SA --iterations 3000 --sa_t 10000000000.0 --sa_c 0.60
jython nn_experiments.py --oa SA --iterations 3000 --sa_t 10000000000.0 --sa_c 0.75
jython nn_experiments.py --oa SA --iterations 3000 --sa_t 10000000000.0 --sa_c 0.90

# vary starting temperature with cooling rate set at 0.15
jython nn_experiments.py --oa SA --iterations 3000 --sa_t 10000000000.0 --sa_c 0.15
jython nn_experiments.py --oa SA --iterations 3000 --sa_t 1000000000.0 --sa_c 0.15
jython nn_experiments.py --oa SA --iterations 3000 --sa_t 100000000.0 --sa_c 0.15
jython nn_experiments.py --oa SA --iterations 3000 --sa_t 10000000.0 --sa_c 0.15
jython nn_experiments.py --oa SA --iterations 3000 --sa_t 1000000.0 --sa_c 0.15