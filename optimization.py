from hashlib import algorithms_available
import numpy as np
import getpass
import os
import platform
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.factory import get_termination
from pymoo.factory import get_crossover
from pymoo.factory import get_mutation

class TensionCompressionSpring (ElementwiseProblem):
    
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

class PressureVessel (ElementwiseProblem):
    
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

class SpeedReducer (ElementwiseProblem):
    
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

def graph_plot (res_algorithm):
    pass

def menu ():
    print('Hello', getpass.getuser(), '!')
    print('Welcome to the mechanical studies design optimization tool\n')
    
    print('Team: ')
    print(' 1. Diego Garzaro: GRR20172364')
    print(' 2. Éder Hamasaki: GRR20172189')
    print(' 3. Leonardo Bein: GRR20172158')
    print(' 4. Natália Choma: GRR20160059')
    print(' 5. Vinícius Parede: GRR20172137\n')

    print('Problems: 1 - Tension/Compression Spring | 2 - Pressure Vessel | 3 - Speed Reducer\n')
    print('The problems are resolved with 3 different algorithmics')
    print('Genetic Algorithmic (GA), Non-dominated Sorting Genetic Algorithm (NSGA-II) and Differential Evolution (DE)\n')

def clean_screen ():
    operating_system = platform.system()
    if operating_system == 'Windows': 
        os.system('cls')
    elif operating_system == 'Linux': 
        os.system('clear')
    else: 
        print(platform.platform(), 'not suported')

def main ():
    clean_screen()
    menu()
    
    problems = [TensionCompressionSpring(), PressureVessel(), SpeedReducer()]
    problems_names = ['Tension/Compression Spring', 'Pressure Vessel', 'Speed Reducer']
    algorithms_names = ['GA', 'NSGA-II', 'DE']

    x_besst_solution = [
        [0.051749, 0.358179, 11.203763],
        [0.8125, 0.4375, 42.098446, 176.636596],
        [3.49999, 0.6999, 17, 7.3, 7.8, 3.3502, 5.2866]
    ]
    y_best_solution = [0.012665, 6059.714339, 2996.3481]

    iterator = 0
    for problem in problems:
        res_algorithm_ga = genetic_algorithm(problem, 0.9, 0.01, 30, False)
        res_algorithm_nsga2 = nsga2_algorithm(problem, 0.9, 0.01, 30, False)
        res_algorithm_de = de_algorithm(problem, 30, False)
        
        print(problems_names[iterator])
        print('------------------------------------------------------')
        print('f(x) = ', y_best_solution[iterator], 'Best Solution')
        print('f(x) = ', res_algorithm_ga.F[0], algorithms_names[0])
        print('f(x) = ', res_algorithm_nsga2.F[0], algorithms_names[1])
        print('f(x) = ', res_algorithm_de.F[0], algorithms_names[2], '\n')

        print('X =', x_besst_solution[iterator], 'Best Solution')
        print('X =', res_algorithm_ga.X, algorithms_names[0])
        print('X =', res_algorithm_nsga2.X, algorithms_names[1])
        print('X =', res_algorithm_de.X, algorithms_names[2])
        print('------------------------------------------------------\n')
        iterator += 1

    '''
    problem = SpeedReducer()

    res_algorithm = de_algorithm(problem, 30, True)

    pop_algorithm = res_algorithm.pop

    print(res_algorithm.X)
    print(res_algorithm.F)

    plt.plot(pop_algorithm.get('F'), color='red')
    plt.show()
    '''

if __name__ == '__main__':
    main()

# problem
# TensionCompressionSpring()
# PressureVessel()
# SpeedReducer()

# algorithm
# genetic_algorithm(problem, 0.9, 0.01, 30, True)
# nsga2_algorithm(problem, 0.9, 0.01, 30, True)
# de_algorithm(problem, 30, True)