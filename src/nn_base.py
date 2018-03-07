"""
Implementation of backpropagation, randomized hill climbing, simulated
annealing, and genetic algorithm to find optimal weights for a neural network
that is classifying the seismic-bumps dataset.

"""
import csv
import time

from helpers import get_abspath

from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem

from func.nn.backprop import RPROPUpdateRule, BatchBackPropagationTrainer
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm


class NNExperiment(object):
    """Neural network experiment object. Runs experiments using various
    random optimization algorithms using specified parameters.

    Args:
        iLayer (int): Number of neurons in input layer.
        hLayer (int): Number of neurons in hidden layer.
        oLayer (int): Number of neurons in output layer.
        iterations (int): Number of iterations.
        oaName (str): Optimization algorithm name.
        measure (AbstractErrorMeasure): Loss function.
        network (BackPropagationNetwork): Neural network object.
        SA_T (float): Temperature (simulated annealing only).
        SA_C (float): Cooling rate (simulated annealing only).
        GA_P (float): Population size (genetic algorithms only).
        GA_MA (float): # of population to mate (genetic algorithms only).
        GA_MU (float): # of population to mutate (genetic algorithms only).

    """

    def __init__(self, iLayer, hLayer_one, hLayer_two, oLayer, iterations, oaName, SA_T=1E10, SA_C=0.10, GA_P=50, GA_MA=10, GA_MU=10):
        self.input_layer = iLayer
        self.hidden_layer_one = hLayer_one
        self.hidden_layer_two = hLayer_two
        self.output_layer = oLayer
        self.iterations = iterations
        self.oaName = oaName
        self.measure = SumOfSquaresError()
        self.network = None
        self.SA_T = SA_T
        self.SA_C = SA_C
        self.GA_P = GA_P
        self.GA_MA = GA_MA
        self.GA_MU = GA_MU

    def get_error(self, network, dataset, measure):
        """Measures the mean squared error (MSE) and classification accuracy
        for a given neural network on a given dataset.

        Args:
            network (BackPropagationNetwork): Neural network object.
            dataset (DataSet): Target dataset.
            measure (AbstractErrorMeasure): Loss function.
        Returns:
            mse (float): Mean-squared error.
            acc (float): Accuracy.

        """
        N = len(dataset)  # number of instances
        error = 0.0
        correct = 0
        incorrect = 0

        # calculate error for each instance in the dataset
        for instance in dataset:
            network.setInputValues(instance.getData())
            network.run()
            actual = instance.getLabel().getContinuous()
            predicted = network.getOutputValues().get(0)
            predicted = max(min(predicted, 1), 0)
            if abs(predicted - actual) < 0.5:
                correct += 1
            else:
                incorrect += 1
            output = instance.getLabel()
            output_values = network.getOutputValues()
            example = Instance(output_values, Instance(output_values.get(0)))
            error += measure.value(output, example)

        # calculate MSE and accuracy
        mse = error / float(N)
        acc = correct / float(correct + incorrect)
        return mse, acc

    def train(self, oa, train, test, validation, filepath):
        """Train a given network on a set of instances.

        Args:
            train (list): List of training instances.
            test (list): List of test instances.
            validation (list): List of validation instances.
            filepath (str): Output file name and path.

        """

        times = [0]
        for iteration in xrange(self.iterations):
            start = time.clock()  # start timer
            oa.train()  # train network
            elapsed = time.clock() - start  # training time
            times.append(times[-1] + elapsed)

            # record MSE, accuracy, and elapsed time every 10 iterations
            if iteration % 10 == 0:
                MSE_train, acc_train = self.get_error(
                    self.network, train, self.measure)
                MSE_test, acc_test = self.get_error(
                    self.network, test, self.measure)
                MSE_valid, acc_valid = self.get_error(
                    self.network, validation, self.measure)
                res = '{},{},{},{},{},{},{},{}\n'.format(
                    iteration, MSE_train, MSE_valid, MSE_test, acc_train, acc_valid, acc_test, times[-1])
                with open(filepath, 'a+') as f:
                    f.write(res)

    def run_experiment(self, train, test, validation):
        """Run experiment

        Args:
            train (list): List of training instances.
            test (list): List of test instances.
            validation (list): List of validation instances.

        """
        factory = BackPropagationNetworkFactory()  # instantiate main NN class
        params = [self.input_layer, self.hidden_layer_one, self.hidden_layer_two, self.output_layer]
        self.network = factory.createClassificationNetwork(params)
        dataset = DataSet(train)  # setup training instances dataset
        nnop = NeuralNetworkOptimizationProblem(
            dataset, self.network, self.measure)
        oa = None

        # get output file name
        outpath = 'results/NN'
        filename = None

        # options for different optimization algorithms
        if self.oaName == 'BP':
            filename = '{}/results.csv'.format(self.oaName)
            rule = RPROPUpdateRule()
            oa = BatchBackPropagationTrainer(
                dataset, self.network, self.measure, rule)
        elif self.oaName == 'RHC':
            filename = '{}/results.csv'.format(self.oaName)
            oa = RandomizedHillClimbing(nnop)
        elif self.oaName == 'SA':
            filename = '{}/results_{}_{}.csv'.format(
                self.oaName, self.SA_T, self.SA_C)
            oa = SimulatedAnnealing(self.SA_T, self.SA_C, nnop)
        elif self.oaName == 'GA':
            filename = '{}/results_{}_{}_{}.csv'.format(
                self.oaName, self.GA_P, self.GA_MA, self.GA_MU)
            oa = StandardGeneticAlgorithm(
                self.GA_P, self.GA_MA, self.GA_MU, nnop)

        # train network
        filepath = get_abspath(filename, outpath)
        self.train(oa, train, test, validation, filepath)


def initialize_instances(input_file):
    """Read a dataset into a list of instances compatible the ABAGAIL NN.
    Assumes that the class labels are 0 or 1.

    Args:
        input_file (str): Input file with classes and attributes.
    Returns:
        instances (list): List of instances (attribute/class value pairs)

    """
    instances = []

    # read in the input file
    with open(input_file, "r") as dataset:
        reader = csv.reader(dataset)
        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            instance.setLabel(Instance(0 if float(row[-1]) <= 0 else 1))
            instances.append(instance)

    return instances
