import os
import time
from array import array
from helpers import get_abspath
from java.util import Random
from opt import SwapNeighbor, DiscreteChangeOneNeighbor
from opt import RandomizedHillClimbing, GenericHillClimbingProblem
from opt import SimulatedAnnealing
from opt.example import TravelingSalesmanRouteEvaluationFunction
from opt.example import TravelingSalesmanSortEvaluationFunction
from opt.example import FlipFlopEvaluationFunction
from opt.example import ContinuousPeaksEvaluationFunction
from opt.example import TravelingSalesmanCrossOver
from opt.ga import GenericGeneticAlgorithmProblem, StandardGeneticAlgorithm
from opt.ga import SingleCrossOver, SwapMutation, DiscreteChangeOneMutation
from opt.prob import GenericProbabilisticOptimizationProblem, MIMIC
from dist import DiscreteUniformDistribution, DiscreteDependencyTree
from shared import FixedIterationTrainer


class TravellingSalesmanOP(object):
    """Travelling salesman optimization problem.

    Args:
        N (int): Number of points in the route.
        subtype (str): Sort or route-based.

    """
    def __init__(self, N, subtype='route'):
        self.N = N
        self.subtype = subtype

    def get_ef(self):
        """Creates a new travelling salesman route evaluation function with
        the specified class variables.

        Returns
            ranges (array): Array of values as specified by N.
            ef (TravelingSalesmanEvaluationFunction): Evaluation function.

        """
        random = Random()
        points = [[0 for x in xrange(2)] for x in xrange(self.N)]
        for i in range(0, len(points)):
            points[i][0] = random.nextDouble()
            points[i][1] = random.nextDouble()

        # create ranges
        fill = [self.N] * self.N
        ranges = array('i', fill)

        if self.subtype == 'route':
            return ranges, TravelingSalesmanRouteEvaluationFunction(points)
        elif self.subtype == 'sort':
            return ranges, TravelingSalesmanSortEvaluationFunction(points)


class FlipFlopOP(object):
    """Flip flop optimization problem.

    Args:
        N (int): Number of elements.

    """
    def __init__(self, N):
        self.N = N

    def get_ef(self):
        """Creates a new flip flop problem with the specified class variables.

        Returns:
            ranges (array): Array of values as specified by N.
            ef (FlipFlopEvaluationFunction): Evaluation function.

        """
        fill = [2] * self.N
        ranges = array('i', fill)

        return ranges, FlipFlopEvaluationFunction()


class ContinuousPeaksOP(object):
    """Flip flop optimization problem.

    Args:
        N (int): Number of elements.
        T (int): Reward threshold for contiguous bits.

    """
    def __init__(self, N, T):
        self.N = N
        self.T = T

    def get_ef(self):
        """Creates a new continuous peaks problem with the specified
        parameters.

        Returns:
            ranges (array): Array of values as specified by N.
            ef (ContinuousPeaksEvaluationFunction): Evaluation function.

        """
        fill = [2] * self.N
        ranges = array('i', fill)

        return ranges, ContinuousPeaksEvaluationFunction(self.T)


class RHCExperiment(object):
    """Creates a randomized hill climbing experiment object.

    Args:
        op (AbstractOptimizationProblem): Optimization problem object.
        trials (int): Number of trials to run.
        iterations (int): Number of iterations to optimize over.

    """
    def __init__(self, op, numTrials, maxIters):
        self.op = op
        self.numTrials = numTrials
        self.maxIters = maxIters

    def run_experiment(self, opName='TSP'):
        """Run a randomized hill climbing optimization experiment for a given
        optimization problem.

        Args:
            ef (AbstractEvaluationFunction): Evaluation function.
            ranges (array): Search space ranges.
            op (str): Name of optimization problem.

        """
        outdir = 'results/OPT/{}'.format(opName)  # get results directory
        fname = get_abspath('RHC_results.csv', outdir)  # get output filename

        # delete existing results file, if it already exists
        try:
            os.remove(fname)
        except Exception as e:
            print e
            pass

        with open(fname, 'w') as f:
            f.write('iterations,fitness,time,fevals,trial\n')

        # start experiment
        for t in range(self.numTrials):
            # initialize optimization problem and training functions
            ranges, ef = self.op.get_ef()
            nf = None
            if opName == 'TSP':
                nf = SwapNeighbor()
            else:
                nf = DiscreteChangeOneNeighbor(ranges)
            odd = DiscreteUniformDistribution(ranges)
            hcp = GenericHillClimbingProblem(ef, odd, nf)
            rhc = RandomizedHillClimbing(hcp)
            fit = FixedIterationTrainer(rhc, 10)

            # run experiment and train evaluation function
            start = time.clock()
            for i in range(0, self.maxIters, 10):
                fit.train()
                elapsed = time.clock() - start
                fe = ef.valueCallCount
                score = ef.value(rhc.getOptimal())
                ef.valueCallCount -= 1

                # write results to output file
                s = '{},{},{},{},{}\n'.format(i + 10, score, fe, elapsed, t)
                with open(fname, 'a+') as f:
                    f.write(s)


class SAExperiment(object):
    """Creates a simulated annealing experiment object.

    Args:
        op (AbstractOptimizationProblem): Optimization problem object.
        trials (int): Number of trials to run.
        iterations (int): Number of iterations to optimize over.
        cr (float): Cooling rate.

    """
    def __init__(self, op, numTrials, maxIters, cr):
        self.op = op
        self.numTrials = numTrials
        self.maxIters = maxIters
        self.cr = cr

    def run_experiment(self, opName):
        """Run a simulated annealing optimization experiment for a given
        optimization problem.

        Args:
            ef (AbstractEvaluationFunction): Evaluation function.
            ranges (array): Search space ranges.
            op (str): Name of optimization problem.

        """
        outdir = 'results/OPT/{}'.format(opName)  # get results directory
        outfile = 'SA_{}_results.csv'.format(self.cr)
        fname = get_abspath(outfile, outdir)  # get output filename

        # delete existing results file, if it already exists
        try:
            os.remove(fname)
        except Exception as e:
            print e
            pass

        with open(fname, 'w') as f:
            f.write('iterations,fitness,time,fevals,trial\n')

        # start experiment
        for t in range(self.numTrials):
            # initialize optimization problem and training functions
            ranges, ef = self.op.get_ef()
            nf = None
            if opName == 'TSP':
                nf = SwapNeighbor()
            else:
                nf = DiscreteChangeOneNeighbor(ranges)
            odd = DiscreteUniformDistribution(ranges)
            hcp = GenericHillClimbingProblem(ef, odd, nf)
            sa = SimulatedAnnealing(1E10, self.cr, hcp)
            fit = FixedIterationTrainer(sa, 10)

            # run experiment and train evaluation function
            start = time.clock()
            for i in range(0, self.maxIters, 10):
                fit.train()
                elapsed = time.clock() - start
                fe = ef.valueCallCount
                score = ef.value(sa.getOptimal())
                ef.valueCallCount -= 1

                # write results to output file
                s = '{},{},{},{},{}\n'.format(i + 10, score, fe, elapsed, t)
                with open(fname, 'a+') as f:
                    f.write(s)


class GAExperiment(object):
    """Creates a genetic algorithms experiment object.

    Args:
        op (AbstractOptimizationProblem): Optimization problem object.
        trials (int): Number of trials to run.
        iterations (int): Number of iterations to optimize over.
        ga_p (int): Population.
        ga_ma (int): Mating rate.
        ga_mu (int): Mutation rate.

    """
    def __init__(self, op, numTrials, maxIters, p, ma, mu):
        self.op = op
        self.numTrials = numTrials
        self.maxIters = maxIters
        self.p = p
        self.ma = ma
        self.mu = mu

    def run_experiment(self, opName):
        """Run a genetic algorithms optimization experiment for a given
        optimization problem.

        Args:
            ef (AbstractEvaluationFunction): Evaluation function.
            ranges (array): Search space ranges.
            op (str): Name of optimization problem.

        """
        outdir = 'results/OPT/{}'.format(opName)  # get results directory
        outfile = 'GA_{}_{}_{}_results.csv'.format(self.p, self.ma, self.mu)
        fname = get_abspath(outfile, outdir)  # get output filename

        # delete existing results file, if it already exists
        try:
            os.remove(fname)
        except Exception as e:
            print e
            pass

        with open(fname, 'w') as f:
            f.write('iterations,fitness,time,fevals,trial\n')

        # start experiment
        for t in range(self.numTrials):
            # initialize optimization problem and training functions
            ranges, ef = self.op.get_ef()
            mf = None
            cf = None
            if opName == 'TSP':
                mf = SwapMutation()
                cf = TravelingSalesmanCrossOver(ef)
            else:
                mf = DiscreteChangeOneMutation(ranges)
                cf = SingleCrossOver()
            odd = DiscreteUniformDistribution(ranges)
            gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
            ga = StandardGeneticAlgorithm(self.p, self.ma, self.mu, gap)
            fit = FixedIterationTrainer(ga, 10)

            # run experiment and train evaluation function
            start = time.clock()
            for i in range(0, self.maxIters, 10):
                fit.train()
                elapsed = time.clock() - start
                fe = ef.valueCallCount
                score = ef.value(ga.getOptimal())
                ef.valueCallCount -= 1

                # write results to output file
                s = '{},{},{},{},{}\n'.format(i + 10, score, fe, elapsed, t)
                with open(fname, 'a+') as f:
                    f.write(s)


class MIMICExperiment(object):
    """Creates a MIMIC experiment object.

    Args:
        op (AbstractOptimizationProblem): Optimization problem object.
        trials (int): Number of trials to run.
        iterations (int): Number of iterations to optimize over.
        s (int): Samples to generate.
        k (float): Samples to keep.
        m (float): Mutation rate.

    """
    def __init__(self, op, numTrials, maxIters, s, k, m):
        self.op = op
        self.numTrials = numTrials
        self.maxIters = maxIters
        self.s = s
        self.k = k
        self.m = m

    def run_experiment(self, opName):
        """Run a MIMIC optimization experiment for a given optimization
        problem.

        Args:
            ef (AbstractEvaluationFunction): Evaluation function.
            ranges (array): Search space ranges.
            op (str): Name of optimization problem.

        """
        outdir = 'results/OPT/{}'.format(opName)  # get results directory
        outfile = 'MIMIC_{}_{}_{}_results.csv'.format(self.s, self.k, self.m)
        fname = get_abspath(outfile, outdir)  # get output filename

        # delete existing results file, if it already exists
        try:
            os.remove(fname)
        except Exception as e:
            print e
            pass

        with open(fname, 'w') as f:
            f.write('iterations,fitness,time,fevals,trial\n')

        # start experiment
        for t in range(self.numTrials):
            # initialize optimization problem and training functions
            ranges, ef = self.op.get_ef()
            mimic = None
            df = DiscreteDependencyTree(self.m, ranges)
            odd = DiscreteUniformDistribution(ranges)
            pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
            mimic = MIMIC(self.s, self.k, pop)
            fit = FixedIterationTrainer(mimic, 10)

            # run experiment and train evaluation function
            start = time.clock()
            for i in range(0, self.maxIters, 10):
                fit.train()
                elapsed = time.clock() - start
                fe = ef.valueCallCount
                score = ef.value(mimic.getOptimal())
                ef.valueCallCount -= 1

                # write results to output file
                s = '{},{},{},{},{}\n'.format(i + 10, score, fe, elapsed, t)
                with open(fname, 'a+') as f:
                    f.write(s)
