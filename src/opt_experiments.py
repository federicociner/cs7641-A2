"""
Run optimization problem experiments using command line utility.

"""
import click
import time
from opt_base import TravellingSalesmanOP, FlipFlopOP, ContinuousPeaksOP
from opt_base import RHCExperiment, SAExperiment, GAExperiment
from opt_base import MIMICExperiment


@click.command()
@click.option('--oa', default='RHC', help='Optimization algorithm name.')
@click.option('--tsp_n', default=75, help='TSP - Number of cities.')
@click.option('--ff_n', default=500, help='FF - Number of elements.')
@click.option('--cp_n', default=100, help='CP - Number of elements.')
@click.option('--cp_t', default=65, help='CP - Reward threshold.')
@click.option('--sa_cr', default=0.15, help='SA - Cooling rate.')
@click.option('--ga_p', default=50, help='GA - population.')
@click.option('--ga_ma', default=10, help='GA - mating rate.')
@click.option('--ga_mu', default=10, help='GA - mutation rate.')
@click.option('--ms', default=100, help='MIMIC - Samples.')
@click.option('--mk', default=50, help='MIMIC - # Samples to keep.')
@click.option('--mm', default=0.3, help='MIMIC - m.')
def run(oa, tsp_n, ff_n, cp_n, cp_t, sa_cr, ga_p, ga_ma, ga_mu, ms, mk, mm):
    # set experiment parameters
    iters = 5000
    trials = 3

    # initialize optimization problem
    tsp = TravellingSalesmanOP(N=tsp_n, subtype='route')
    tspMimic = TravellingSalesmanOP(N=tsp_n, subtype='sort')
    ff = FlipFlopOP(N=ff_n)
    cp = ContinuousPeaksOP(N=cp_n, T=cp_t)

    # set up experiment
    if oa == 'RHC':
        start = time.clock()
        RHC = RHCExperiment(op=tsp, numTrials=trials, maxIters=iters)
        RHC.run_experiment(opName='TSP')

        RHC = RHCExperiment(op=ff, numTrials=trials, maxIters=iters)
        RHC.run_experiment(opName='FF')

        RHC = RHCExperiment(op=cp, numTrials=trials, maxIters=iters)
        RHC.run_experiment(opName='CP')

        elapsed = time.clock() - start
        print('RHC ran in {} seconds'.format(elapsed))
    elif oa == 'SA':
        start = time.clock()
        SA = SAExperiment(op=tsp, numTrials=trials, maxIters=iters, cr=sa_cr)
        SA.run_experiment(opName='TSP')

        SA = SAExperiment(op=ff, numTrials=trials, maxIters=iters, cr=sa_cr)
        SA.run_experiment(opName='FF')

        SA = SAExperiment(op=cp, numTrials=trials, maxIters=iters, cr=sa_cr)
        SA.run_experiment(opName='CP')

        elapsed = time.clock() - start
        print('SA ran with CR {} ran in {} seconds'.format(sa_cr, elapsed))
    elif oa == 'GA':
        start = time.clock()
        GA = GAExperiment(op=tsp, numTrials=trials,
                          maxIters=iters, p=ga_p, ma=ga_ma, mu=ga_mu)
        GA.run_experiment(opName='TSP')

        GA = GAExperiment(op=ff, numTrials=trials,
                          maxIters=iters, p=ga_p, ma=ga_ma, mu=ga_mu)
        GA.run_experiment(opName='FF')

        GA = GAExperiment(op=cp, numTrials=trials,
                          maxIters=iters, p=ga_p, ma=ga_ma, mu=ga_mu)
        GA.run_experiment(opName='CP')

        elapsed = time.clock() - start
        print('GA with population {}, mating rate {} and mutation rate {} ran in {} seconds'.format(ga_p, ga_ma, ga_mu, elapsed))
    elif oa == 'MIMIC':
        start = time.clock()
        MIMIC = MIMICExperiment(op=tspMimic, numTrials=trials,
                                maxIters=iters, s=ms, k=mk, m=mm)
        MIMIC.run_experiment(opName='TSP')

        MIMIC = MIMICExperiment(op=ff, numTrials=trials,
                                maxIters=iters, s=ms, k=mk, m=mm)
        MIMIC.run_experiment(opName='FF')

        MIMIC = MIMICExperiment(op=cp, numTrials=trials,
                                maxIters=iters, s=ms, k=mk, m=mm)
        MIMIC.run_experiment(opName='CP')

        elapsed = time.clock() - start
        print('MIMIC with samples {}, keep {} and m {} ran in {} seconds'.format(ms, mk, mm, elapsed))


if __name__ == '__main__':
    # Run experiments
    run()
