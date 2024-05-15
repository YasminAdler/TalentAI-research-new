import random
from statistic_regular_algo.KMeanClusterer import KMeansClusterer

# Define the search space for each parameter

# Define the size of the population
population_size = 2

# Define the maximum number of generations
max_generations = 1 # changed it from 8 

# Generate an initial population of solutions
def generate_population(beta_space, gamma_space, z, file):
    population = []
    for i in range(population_size):
        t = random.uniform(*[0, z])
        t2 = random.uniform(*[0, z])
        theta1 = min(t, t2)  # random.uniform(*theta1_space)
        theta2 = max(t, t2)  # random.uniform(*[theta1, z])
        beta = random.uniform(*beta_space)
        gamma = random.uniform(*gamma_space)
        solution = (theta1, theta2, beta, gamma)
        population.append(solution)
    return population


# Evaluate the fitness of each solution
def evaluate_population(params, vectors, population, distance_function, type_values, k, file):
    fitness_scores = []
    i = 0
    for solution in population:
        i += 1
        params["theta1"] = solution[0]
        params["theta2"] = solution[1]  # 10
        params["betha"] = solution[2]  # 0.05
        params["gamma"] = solution[3]  # 0.01

        model_for_population = KMeansClusterer(hyper_params=params, distance=distance_function, num_means=k,
                                               type_of_fields=type_values)
        # activate model
        model_for_population.cluster_vectorspace(vectors)
        fitness_scores.append(model_for_population.get_wcss())

    return fitness_scores


def find_lowest_indices(file, numbers):
    indexed_list = list(enumerate(numbers))
    sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1])
    middle_index = len(sorted_indexed_list) // 2
    lowest_half_indexes = [index for index, _ in sorted_indexed_list[:middle_index]]

    return lowest_half_indexes

# Select parents for reproduction
def select_parents(population, fitness_scores, file):
    selected_parents = []
    lowest_scores = find_lowest_indices(file, fitness_scores)

    length = len(lowest_scores) if len(lowest_scores) % 2 == 0 else len(lowest_scores) - 1
    for i in range(0, length, 2):

        parent1 = population[lowest_scores[i]]
        parent2 = population[lowest_scores[i + 1]]

        selected_parents.append((parent1, parent2))
    return selected_parents


# Apply genetic operators to create a new generation
def apply_genetic_operators(selected_parents, beta_space, gamma_space, z, file):
    new_population = []
    for parent1, parent2 in selected_parents:
        # Crossover operator
        new_population.append(parent1)
        new_population.append(parent2)

        partition_index = random.randint(1, len(parent1) - 1)
        child1 = parent1[:partition_index] + parent2[partition_index:]

        partition_index = random.randint(1, len(parent1) - 1)
        child2 = parent2[:partition_index] + parent1[partition_index:]

        # Mutation operator for first child
        if random.random() < 0.1:
            theta1 = random.uniform(*[0, child1[1]])
            child1 = (theta1, child1[1], child1[2], child1[3])
        if random.random() < 0.1:
            theta2 = random.uniform(*[child1[0], z])
            child1 = (child1[0], theta2, child1[2], child1[3])
        if random.random() < 0.1:
            beta = random.uniform(*beta_space)
            child1 = (child1[0], child1[1], beta, child1[3])

        if random.random() < 0.1:
            gamma = random.uniform(*gamma_space)
            child1 = (child1[0], child1[1], child1[2], gamma)

        # mutation operator for second child
        if random.random() < 0.1:
            theta1 = random.uniform(*[0, child2[1]])
            child2 = (theta1, child2[1], child2[2], child2[3])
        if random.random() < 0.1:
            theta2 = random.uniform(*[child2[0], z])
            child2 = (child2[0], theta2, child2[2], child2[3])
        if random.random() < 0.1:
            beta = random.uniform(*beta_space)
            child2 = (child2[0], child2[1], beta, child2[3])

        if random.random() < 0.1:
            gamma = random.uniform(*gamma_space)
            child2 = (child2[0], child2[1], child2[2], gamma)



        new_population.append(child1)
        new_population.append(child2)

    return new_population


def genetic_algorithm(params, distance_function, k, vectors, type_values, z):

    # Open the file for writing
    # If the file doesn't exist, it will be created. If it does exist, its contents will be overwritten.
    file = open('../logger.txt', 'a')

    if distance_function.__name__ != "Statistic":
        print("genetic_algorithm no matter")

        return (0, 0, 0, 0)


    z = z

    beta_space = [0, 1]
    gamma_space = [0, 1]

    # Generate an initial population
    population = generate_population(beta_space, gamma_space, z, file)

    # Repeat the genetic algorithm for a maximum of max_generations

    for generation in range(max_generations):
        file.write(f'\n the {generation}th generation \n')
        # print(generation, "###############out of", max_generations)

        # Evaluate the fitness of the current population
        fitness_scores = evaluate_population(params, vectors, population, distance_function, type_values, k, file)

        # Select parents for reproduction
        selected_parents = select_parents(population, fitness_scores, file)

        # Apply genetic operators to create a new generation
        population = apply_genetic_operators(selected_parents, beta_space, gamma_space, z, file)
    # Find the best solution in the final population
    best_solution = population[0]

    params["theta1"] = best_solution[0]
    params["theta2"] = best_solution[1]  # 10
    params["betha"] = best_solution[2]  # 0.05
    params["gamma"] = best_solution[3]  # 0.01

    model_for_population = KMeansClusterer(hyper_params=params, distance=distance_function, num_means=k,
                                           type_of_fields=type_values)

    # activate model
    model_for_population.cluster_vectorspace(vectors)
    # best_fitness_score = model_for_population.wcss_calculate()  # hello(*best_solution)
    best_fitness_score = model_for_population.get_wcss()  # hello(*best_solution)

    for solution in population[1:]:

        params["theta1"] = solution[0]
        params["theta2"] = solution[1]  # 10
        params["betha"] = solution[2]  # 0.05
        params["gamma"] = solution[3]  # 0.01

        model_for_population = KMeansClusterer(hyper_params=params, distance=distance_function,
                                               num_means=k, type_of_fields=type_values)
        # activate model
        model_for_population.cluster_vectorspace(vectors)

        fitness_score = model_for_population.get_wcss()  # hello(*solution)
        # if we want to find the biggest score
        if fitness_score < best_fitness_score:
            best_solution = solution
            best_fitness_score = fitness_score

    best_solution = best_solution  # Replace this with your actual best_solution value


    return best_solution  # , best_fitness_score
