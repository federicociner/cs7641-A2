### CS7641 Assignment 2
### Federico Ciner (fciner3)
### Spring 2018

This project contains all of the necessary code and data to run the experiments used in CS741 Assignment 2.

## Setup

1. Ensure you have Python 2.7, Jython 2.7.0, Java 8 and ABAGAIL installed and configured on your CLASS path. Alternatively, you can use the Docker image included in this repository or the "federicociner/cs7641-a2" Docker image on DockerHub.

2. Run "pip install -r requirements.txt" to install all the required CPython dependencies.

3. To use the command line functions to run the experiments, you will have to install the "click" Python package in Jython using "jython easy-install click."

## Running the experiments

1. To run the experiments for the neural networks weight optimization (part 1), you can run the following Shell scripts to generate the output CSVs in the "results" directory (including grid search of hyperparameters for some of the algorithms):
    - run_bp.sh
    - run_ga.sh
    - run_rhc.sh
    - run_sa.sh

2. In order to generate plots for part 1, run the nn_plots.py file using CPython, which will output PNG plot files to the "plots" directory.

3. To run the experiments and generate plots for the three optimization problems (part 2), you can simply run the run_opt.sh Shell script, which will generate the output CSVs and PNG plot files for each of the three optimization problems in the "plots" and "results" directories.

## Credits and References
All implementations and experiment code for this assignment were taken from the ABAGAIL repository (https://github.com/pushkar/ABAGAIL) and the Jython code was adapted from Jonathan Tay's repository (https://github.com/JonathanTay/CS-7641-assignment-2).