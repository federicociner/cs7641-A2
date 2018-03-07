from array import array
from helpers import get_abspath
from opt.example import TravelingSalesmanRouteEvaluationFunction
from opt.example import TravelingSalesmanSortEvaluationFunction
from opt.example import FlipFlopEvaluationFunction
from opt.example import ContinuousPeaksEvaluationFunction
from java.util import Random

import sys
import os
import time
from time import clock
from itertools import product

from dist import DiscreteUniformDistribution
from dist import DiscreteDependencyTree
from dist import Distribution
from dist import DiscretePermutationDistribution

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean

import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
from opt.prob import ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import shared.Instance as Instance
import util.ABAGAILArrays as ABAGAILArrays


class OptimizationProblem(object):
    """Optimization problem object. Creates an optimization problem and
    returns the evaluation function and any other relevant variables.

    """

    def tsp(self, N=20, type='route'):
        """Creates a new travelling salesman route problem with the specified
        parameters.

        Args
            N (int): Number of "locations" or points in the route.
            type (str): Determines if evaluation function is sort or
            route-based.
        Returns
            ef (TravelingSalesmanEvaluationFunction): Evaluation function.

        """
        random = Random()
        points = [[0 for x in xrange(2)] for x in xrange(N)]
        for i in range(0, len(points)):
            points[i][0] = random.nextDouble()
            points[i][1] = random.nextDouble()

        # create ranges
        fill = [N] * N
        ranges = array('i', fill)

        if type == 'route':
            return TravelingSalesmanRouteEvaluationFunction(points), ranges
        elif type == 'sort':
            return TravelingSalesmanSortEvaluationFunction(points), ranges

    def flipflop(self, N=50):
        """Creates a new flip flop problem with the specified parameters

        Args:
            N (int): Number of binary elements.
        Returns:
            ranges (array): Array of values as specified by N.
            ef (FlipFlopEvaluationFunction): Evaluation function.

        """
        fill = [2] * N
        ranges = array('i', fill)

        return ranges, FlipFlopEvaluationFunction()

    def continuouspeaks(self, T=20, N=100):
        """Creates a new continuous peaks problem with the specified
        parameters.

        Args:
            N (int): Number of binary elements.
            T (int): Reward threshold for contiguous bits.
        Returns:
            ranges (array): Array of values as specifieed by N.
            ef (ContinuousPeaksEvaluationFunction): Evaluation function.

        """
        fill = [2] * N
        ranges = array('i', fill)

        return ranges, ContinuousPeaksEvaluationFunction(T)


class RHCExperiment(object):
    """Creates a randomized hill climbing experiment object.

    Args:
        trials (int): Number of trials to run.
        iterations (int): Number of iterations to optimize over.

    """
    def __init__(self, trials, iterations):
        self.trials = trials
        self.iterations = iterations

    def run_experiment(self, ef, ranges):







class MIMICExperiment(object):
    """Creates a MIMIC experiment object.

    """


class SAExperiment(object):
    """Creates a simulated annealing experiment object.

    """


class GAExperiment(object):
    """Creates a genetic algorithm experiment object.

    """

    outfile = './TSP/TSP_@ALG@_@N@_LOG.txt'
    ef = TravelingSalesmanRouteEvaluationFunction(points)
    odd = DiscretePermutationDistribution(N)
    nf = SwapNeighbor()
    mf = SwapMutation()
    cf = TravelingSalesmanCrossOver(ef)
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)

    MIMIC - TSP
    ef = TravelingSalesmanSortEvaluationFunction(points)
    odd = DiscreteUniformDistribution(ranges)

    for t in range(numTrials):
        for samples, keep, m in product([100], [50], [0.1, 0.3, 0.5, 0.7, 0.9]):
            fname = outfile.replace('@ALG@', 'MIMIC{}_{}_{}'.format(
                samples, keep, m)).replace('@N@', str(t + 1))
            df = DiscreteDependencyTree(m, ranges)
            with open(fname, 'w') as f:
                f.write('iterations,fitness,time,fevals\n')
            ef = TravelingSalesmanSortEvaluationFunction(points)
            pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
            mimic = MIMIC(samples, keep, pop)
            fit = FixedIterationTrainer(mimic, 10)
            times = [0]
            for i in range(0, maxIters, 10):
                start = clock()
                fit.train()
                elapsed = time.clock() - start
                times.append(times[-1] + elapsed)
                fevals = ef.fevals
                score = ef.value(mimic.getOptimal())
                ef.fevals -= 1
                st = '{},{},{},{}\n'.format(i, score, times[-1], fevals)
                print st
                with open(fname, 'a') as f:
                    f.write(st)

    # SA - TSP
    for t in range(numTrials):
        for CE in [0.15, 0.35, 0.55, 0.75, 0.95]:
            fname = outfile.replace('@ALG@', 'SA{}'.format(
                CE)).replace('@N@', str(t + 1))
            with open(fname, 'w') as f:
                f.write('iterations,fitness,time,fevals\n')
            ef = TravelingSalesmanRouteEvaluationFunction(points)
            hcp = GenericHillClimbingProblem(ef, odd, nf)
            sa = SimulatedAnnealing(1E10, CE, hcp)
            fit = FixedIterationTrainer(sa, 10)
            times = [0]
            for i in range(0, maxIters, 10):
                start = clock()
                fit.train()
                elapsed = time.clock() - start
                times.append(times[-1] + elapsed)
                fevals = ef.fevals
                score = ef.value(sa.getOptimal())
                ef.fevals -= 1
                st = '{},{},{},{}\n'.format(i, score, times[-1], fevals)
                print st
                with open(fname, 'a') as f:
                    f.write(st)

    # GA - TSP
    for t in range(numTrials):
        for pop, mate, mutate in product([100], [50, 30, 10], [50, 30, 10]):
            fname = outfile.replace('@ALG@', 'GA{}_{}_{}'.format(
                pop, mate, mutate)).replace('@N@', str(t + 1))
            with open(fname, 'w') as f:
                f.write('iterations,fitness,time,fevals\n')
            ef = TravelingSalesmanRouteEvaluationFunction(points)
            gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
            ga = StandardGeneticAlgorithm(pop, mate, mutate, gap)
            fit = FixedIterationTrainer(ga, 10)
            times = [0]
            for i in range(0, maxIters, 10):
                start = clock()
                fit.train()
                elapsed = time.clock() - start
                times.append(times[-1] + elapsed)
                fevals = ef.fevals
                score = ef.value(ga.getOptimal())
                ef.fevals -= 1
                st = '{},{},{},{}\n'.format(i, score, times[-1], fevals)
                print st
                with open(fname, 'a') as f:
                    f.write(st)




if __name__ == '__main__':
    # set experiment parameters
    maxIters = 3000
    numTrials = 5

    # test
    OP = OptimizationProblem()
    ranges, ef = OP.flipflop(N=5)
    print ranges
    print ef
