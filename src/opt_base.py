import os
import time
from array import array
from helpers import get_abspath
from java.util import Random
from opt import SwapNeighbor, DiscreteChangeOneNeighbor
from opt import RandomizedHillClimbing, GenericHillClimbingProblem
from opt.example import TravelingSalesmanRouteEvaluationFunction
from opt.example import TravelingSalesmanSortEvaluationFunction
from opt.example import FlipFlopEvaluationFunction
from opt.example import ContinuousPeaksEvaluationFunction
from shared import FixedIterationTrainer
from dist import DiscreteUniformDistribution

import sys
from itertools import product

from dist import DiscreteDependencyTree
from dist import Distribution
from dist import DiscretePermutationDistribution

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean

import opt.EvaluationFunction as EvaluationFunction
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
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
from opt.prob import GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
from opt.prob import ProbabilisticOptimizationProblem
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import shared.Instance as Instance
import util.ABAGAILArrays as ABAGAILArrays


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
            odd = DiscreteUniformDistribution(ranges)  # get ranges
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

        # return res


if __name__ == '__main__':
    # set experiment parameters
    maxIters = 3000
    numTrials = 20

    # initialize optimization problem
    tsp = TravellingSalesmanOP(N=100, subtype='route')
    ff = FlipFlopOP(N=1000)
    cp = ContinuousPeaksOP(N=100, T=49)

    # set up experiment
    RHC = RHCExperiment(op=tsp, numTrials=numTrials, maxIters=maxIters)
    RHC.run_experiment(opName='TSP')

    RHC = RHCExperiment(op=ff, numTrials=numTrials, maxIters=maxIters)
    RHC.run_experiment(opName='FF')

    RHC = RHCExperiment(op=cp, numTrials=numTrials, maxIters=maxIters)
    RHC.run_experiment(opName='CP')
