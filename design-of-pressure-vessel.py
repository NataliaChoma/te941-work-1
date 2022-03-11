import pygad
import numpy

def f(solution):
    x1, x2, x3, x4 = tuple(solution)
    return (0.6224*x1*x3*x4 + 1.7781*x2*(x3**2) + 3.1661*(x1**2)*x4 + 19.84*(x1**2)*x3)

def fitness_function(solution, solution_idx):
    x1, x2, x3, x4 = tuple(solution)

    restrictions = [
        (0.1 <= x1 <= 99) and (0.1 <= x2 <= 99),
        (10 <= x3 <= 200) and (10 <= x4 <= 200),
        (x1 % 0.0625 == 0) and (x2 % 0.0625 == 0),
        # (x1 <= 0.0193 * x3),
        # (x2 <= 0.00954 * x3),
        # (numpy.pi*(x3**2)*x4 +(4/3)*numpy.pi*(x3**3) <= 1296000),
        # (x4 <= 240),
    ]
   

    for restriction in restrictions:
        if not restriction:
            return 0.0

    print("solution")
    print(restrictions)

    return 1/f(solution)


print(fitness_function((0.8125, 0.4375, 42.098446, 176.636596), 1))

num_generations = 5000 # Number of generations.
num_parents_mating = 7 # Number of solutions to be selected as parents in the mating pool.

# To prepare the initial population, there are 2 ways:
# 1) Prepare it yourself and pass it to the initial_population parameter. This way is useful when the user wants to start the genetic algorithm with a custom initial population.
# 2) Assign valid integer values to the sol_per_pop and num_genes parameters. If the initial_population parameter exists, then the sol_per_pop and num_genes parameters are useless.
sol_per_pop = 100 # Number of solutions in the population.
num_genes = 4

last_fitness = 0
def callback_generation(ga_instance):
    global last_fitness
    # print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    # print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    # print("Change     = {change}".format(change=ga_instance.best_solution()[1] - last_fitness))
    # print("Solution     = {change}".format(change=ga_instance.best_solution()[0]))
    last_fitness = ga_instance.best_solution()[1]

# Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
ga_instance = pygad.GA(num_generations=num_generations,
                       gene_type=numpy.float64,
                       num_parents_mating=num_parents_mating,
                    #    mutation_probability=0.5,
                    #    mutation_percent_genes=0.2,
                       crossover_probability=0.90,
                       mutation_probability=0.01, 
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       init_range_low=10,
                       init_range_high=20, 
                       num_genes=num_genes,
                       allow_duplicate_genes=False,
                       on_generation=callback_generation)
print(ga_instance.initial_population)
# Running the GA to optimize the parameters of the function.
ga_instance.run()

# After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
ga_instance.plot_fitness()

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=fitness_function(solution, 1)))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

prediction = f(solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

# # Saving the GA instance.
# filename = 'genetic' # The filename to which the instance is saved. The name is without extension.
# ga_instance.save(filename=filename)

# # Loading the saved GA instance.
# loaded_ga_instance = pygad.load(filename=filename)
# loaded_ga_instance.plot_fitness()