# Work 1: Evolutionary computation

Universidade Federal do Paraná (UFPR)

**Team:** 
1. Diego Garzaro
2. Éder Hamasaki
3. Leonardo Bein
4. Vinícius Parede

## Part 01: Theoretical

### 1) What is evolutionary computation?

Evolutionary Computation (EC) is a sub-field of Computational Intelligence, a branch of Machine Learning and Artificial Intelligence. The applications of Evolutionary Computation are numerous, from solving optimization problems, designing robots, creating decision trees, tuning data mining algorithms, training neural networks, and tuning hyperparameters. Evolutionary computation is a family of algorithms for global optimization inspired by biological evolution.

Evolutionary computation is commonly used instead of standard numerical method when there is no known derivative of the fitness function (reinforcement type learning), or when the fitness function has many local extrema that can trap sequential methods.

EC is a computational intelligence technique inspired from natural evolution. An EC algorithm starts with creating a population consisting of individuals that represent solutions to the problem. The first population could be created randomly or fed into the algorithm. Individuals are evaluated with a fitness function, and the output of the function shows how well this individual solves or comes close to solving the problem.

Then, some operators inspired from natural evolution, such as crossover, mutation, selection, and reproduction, are applied to individuals. Based on the fitness values of newly evolved individuals, a new population is generated. Because the population size has to be preserved as in nature, some individuals are eliminated. This process goes on until the termination criterion is met. Reaching the number of generations defined is the most used criterion to stop the algorithm. The best individual with the highest fitness value is selected as the solution

The general steps of an evolutionary computation algorithm are shown below.
1. initilize population;
2. evaluate the fitness value of each individual;
3. **while** the optimal solution is not found and the number of generations defined is not reached select parents;
4. apply genetic operators to the selected individuals;
5. evaluate fitness values of new individuals;
6. select individuals for the next generation.
7. end **while**
8. **return** the best individual


**References:**
 - [Evolutionary Computation - Medium](https://towardsdatascience.com/evolutionary-computation-full-course-overview-f4e421e945d9)
 - [Evolutionary Computation - Wikipedia](https://en.wikipedia.org/wiki/Evolutionary_computation)
 - [Evolutionary Computation - ScienceDirect](https://www.sciencedirect.com/topics/computer-science/evolutionary-computation)
 - [A Survey of Intrusion Detection Systems Using Evolutionary Computation - ScienceDirect](https://www.sciencedirect.com/science/article/pii/B9780128015384000045) 

### 2) Explain some basic concepts and terms related to genetic algorithm with binary representation i.e. population, chromosome, gene, allele, fitness function, and genetic operators

Text Text Text teste

### 3) Compare the single-point and two-point crossover in a genetic algorithm with binary representation.

Text Text Text

### 4) What are some potentialities and disadvantages of genetic algorithms?

Text Text Text

### 5) Describe the crossover (recombination) operation in a classical differential evolutionapproach.

Text Text Text

### 6) Describe the mutation operation in a classical differential evolution approach.

Text Text Text
