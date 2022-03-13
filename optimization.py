import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
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


# numero_genes * pop_size = 
# 1 -> 30 * iteration
# 2 -> 30 * 2 = 60
# ..
# 10000 

algorithm = NSGA2(
    pop_size=100,
    crossover=get_crossover('real_sbx', prob=0.8),
    mutation=get_mutation('real_pm', prob=0.05)
)   
termination = get_termination("n_gen", 32)

tension_compression = TensionCompressionSpring()
pressure_vessel = PressureVessel()

'''
res = minimize(
    tension_compression,
    algorithm,
    termination,
    seed=1,
    save_history=True,
    verbose=True
)
print(res.X)
print(res.F)
'''
res = minimize(
    pressure_vessel,
    algorithm,
    termination,
    seed=1,
    save_history=True,
    verbose=True
)
pop = res.pop
print(res.X)
print(res.F)

#plt.plot(pop.get('F'))
#plt.show()

