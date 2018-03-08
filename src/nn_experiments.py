"""
Run neural network experiments using command line utility.

"""
import os
from nn_base import initialize_instances, NNExperiment
from helpers import get_abspath
import click


@click.command()
@click.option('--oa', default='BP', help='Optimization algorithm name.')
@click.option('--iterations', default=1000, help='Number of iterations.')
@click.option('--sa_t', default=1E10, help='Temperature (SA only).')
@click.option('--sa_c', default=0.10, help='Cooling rate (SA only).')
@click.option('--ga_p', default=50, help='Population size (GA only).')
@click.option('--ga_ma', default=10, help='# population to mate (GA only).')
@click.option('--ga_mu', default=10, help='# population to mutate (GA only).')
def run(oa, iterations, sa_t, sa_c, ga_p, ga_ma, ga_mu):
    """Run neural network experiment

    """
    # get dataset filepaths
    filepath = 'data/experiments'
    train_file = 'seismic_train.csv'
    test_file = 'seismic_test.csv'
    validation_file = 'seismic_validation.csv'

    # load datasets as instances
    train_ints = initialize_instances(get_abspath(train_file, filepath))
    test_ints = initialize_instances(get_abspath(test_file, filepath))
    valid_ints = initialize_instances(get_abspath(validation_file, filepath))

    # set NN parameters
    input_layer = 21  # number of features
    hidden_layer_one = 250  # hidden layer one nodes
    hidden_layer_two = 250  # hidden layer two nodes
    output_layer = 1  # output layer is always 1 for binomial classification

    # define optimization algorithm
    respath = 'results/NN'
    resfile = None
    if oa in ('BP', 'RHC'):
        resfile = get_abspath('{}/results.csv'.format(oa), respath)
    elif oa == 'SA':
        resfile = get_abspath('{}/results_{}_{}.csv'.format(oa, sa_t, sa_c), respath)
    elif oa == 'GA':
        resfile = get_abspath('{}/results_{}_{}_{}.csv'.format(oa, ga_p, ga_ma, ga_mu), respath)

    # remove existing results file, if it exists
    try:
        os.remove(resfile)
    except Exception as e:
        print e
        pass

    # recreate base results file
    with open(resfile, 'w') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format('iteration'
                                                  ,'MSE_train'
                                                  ,'MSE_validation'
                                                  ,'MSE_test'
                                                  ,'acc_train'
                                                  ,'acc_validation'
                                                  ,'acc_test'
                                                  ,'seconds_elapsed'))

    # initialize experiment
    NN = NNExperiment(input_layer
                     ,hidden_layer_one
                     ,hidden_layer_two
                     ,output_layer
                     ,iterations
                     ,oa
                     ,SA_T=sa_t
                     ,SA_C=sa_c
                     ,GA_P=ga_p
                     ,GA_MA=ga_ma
                     ,GA_MU=ga_mu)
    NN.run_experiment(train_ints, test_ints, valid_ints)


if __name__ == '__main__':
    # run NN experiment
    run()
