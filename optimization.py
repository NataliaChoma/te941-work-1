from xmlrpc.client import boolean
import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_termination
from pymoo.factory import get_crossover
from pymoo.factory import get_mutation

class TensionCompressionSpring(ElementwiseProblem):
    
    # 0.05 <= x[0] <= 2
    # 0.25 <= x[1] <= 1.3
    # 2 <= x[2] <= 15
    def __init__(self):
        super().__init__(n_var=3,
                         n_obj=1,
                         n_constr=4,
                         xl=np.array([0.05, 0.25, 2]),
                         xu=np.array([2, 1.3, 15]))

    def _evaluate(self, x, out, *args, **kwargs):
        # X = (x[0], x[1], x[2]) = (d, D, N)

        # function
        f = (x[2] + 2) * x[1] * x[0] ** 2

        # constraints 
        g1 = 1 - ((x[1] ** 3 * x[2]) / (71785 * x[0] ** 4))
        g2 = ((4 * x[1] ** 2 - x[0] * x[1]) / (12566 * (x[1] * x[0] ** 3 - x[0] ** 4))) + (1 / (5108 * x[0] ** 2)) - 1
        g3 = 1 - ((140.45 * x[0]) / (x[1] ** 2 * x[2]))
        g4 = ((x[1] + x[0]) / 1.5) - 1

        out["F"] = [f]
        out["G"] = [g1, g2, g3, g4]

class PressureVessel(ElementwiseProblem):
    
    # 0.1 <= x[0] <= 99
    # 0.1 <= x[1] <= 99
    # 10 <= x[2] <= 200
    # 10 <= x[3] <= 200
    def __init__(self):
        super().__init__(n_var=4,
                         n_obj=1,
                         n_constr=4,
                         xl=np.array([0.0625, 0.0625, 10, 10]),
                         xu=np.array([6.1875, 6.1875, 200, 200]))

    def _evaluate(self, x, out, *args, **kwargs):
        x[0] = np.round(x[0] / 0.0625, 0) * 0.0625
        x[1] = np.round(x[1] / 0.0625, 0) * 0.0625
        
        # function
        f = 0.6224 * x[0] * x[2] * x[3] + \
            1.7781 * x[1] * (x[2] ** 2) + \
            3.1661 * (x[0] ** 2) * x[3] + \
            19.84*(x[0] ** 2) * x[2]

        # constraints 
        g1 = -x[0] + 0.0193 * x[2]
        g2 = -x[1] + 0.00954 * x[2]
        g3 = -np.round(np.pi, 5) * x[2] ** 2 * x[3] - \
             (4/3) * np.round(np.pi, 5) * x[2] ** 3 + \
             1296000
        g4 = x[3] - 240

        out["F"] = [f]
        out["G"] = [g1, g2, g3, g4]

class SpeedReducer(ElementwiseProblem):
    
    def __init__(self):
        super().__init__(n_var=7,
                         n_obj=1,
                         n_constr=11,
                         xl=np.array([2.6, 0.7, 17, 7.3, 7.8, 2.9, 5.0]),
                         xu=np.array([3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5]))

    def _evaluate(self, x, out, *args, **kwargs):
        x1, x2, x3, x4, x5, x6, x7 = tuple(x)

        # function
        f = 0.7854 * x1 * (x2 ** 2) * \
            ( 3.3333 * (x3 ** 2) + 14.9334 * x3 - 43.0934) - \
            1.508 * x1 * ( x6 ** 2 + x7 ** 2 ) + \
            7.4777 * ( x6 ** 3 + x7 ** 3 ) + \
            0.7854 * (x4 * (x6 ** 2) + x5 * (x7**2) )

        # constraints 
        g1  = 27/(x1 * ( x2 ** 2 ) * x3) - 1
        g2  = 397.5/(x1 * ( x2 ** 2 ) * (x3 ** 2)) - 1
        g3  = (1.93 * ( x4 ** 3 ))/(x2 * x3  * (x6 ** 4)) - 1
        g4  = (1.93 * ( x5 ** 3 ))/(x2 * x3  * (x7 ** 4)) - 1
        g5  = np.sqrt( ( (745*x4)/(x2*x3) )**2 + 16.9e6 )/(110 * (x6 ** 3)) - 1
        g6  = np.sqrt( ( (745*x5)/(x2*x3) )**2 + 157.5e6 )/(85 * (x7 ** 3)) - 1
        g7  = x2*x3/40 - 1
        g8  = (5*x2)/x1 - 1
        g9  = x1/(12*x2) - 1
        g10 = (1.5*x6 + 1.9)/x4 - 1
        g11 = (1.1*x7 + 1.9)/x5 - 1

        out["F"] = [f]
        out["G"] = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11]



def genetic_algorithm (problem: object, crossover_probability: float, mutation_probability: float, number_genes: int, print_verbose: bool):
    termination = get_termination("n_gen", number_genes)
    algorithm_ga = GA(
        pop_size=100,
        crossover=get_crossover('real_sbx', prob=crossover_probability),
        mutation=get_mutation('real_pm', prob=mutation_probability)
    )
    res_ga = minimize(
        problem,
        algorithm_ga,
        termination,
        seed=1,
        save_history=True,
        verbose=print_verbose
    )
    return res_ga

def nsga2_algorithm (problem: object, crossover_probability: float, mutation_probability: float, number_genes: int, print_verbose: bool):
    termination = get_termination("n_gen", number_genes)
    algorithm_nsga2 = NSGA2(
        pop_size=100,
        crossover=get_crossover('real_sbx', prob=crossover_probability),
        mutation=get_mutation('real_pm', prob=mutation_probability)
    )
    res_nsga2 = minimize(
        problem,
        algorithm_nsga2,
        termination,
        seed=1,
        save_history=True,
        verbose=print_verbose
    )
    return res_nsga2

def de_algorithm (problem: object, number_genes: int, print_verbose: bool):
    termination = get_termination("n_gen", number_genes)
    algorithm_de = DE(
        pop_size=100,
        sampling=LHS(),
        variant="DE/rand/1/bin",
        CR=0.3,
        dither="vector",
        jitter=False
    ) 
    res_de = minimize(
        problem,
        algorithm_de,
        termination,
        seed=1,
        save_history=True,
        verbose=print_verbose
    )
    return res_de

# problem
# TensionCompressionSpring()
# PressureVessel()
# SpeedReducer()

# algorithm
# genetic_algorithm(problem, 0.9, 0.01, 30, True)
# nsga2_algorithm(problem, 0.9, 0.01, 30, True)
# de_algorithm(problem, 30, True)

problem = PressureVessel()

res_algorithm = de_algorithm(problem, 30, True)

pop_algorithm = res_algorithm.pop
pop_pop_algorithm = res_algorithm.pop

print(res_algorithm.X)
print(res_algorithm.F)

plt.plot(pop_algorithm.get('F'), color='red')
plt.show()

