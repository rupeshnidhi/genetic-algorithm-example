import numpy as np
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt

bounds = [
    {
        'lower': 0.0,
        'upper': 25.0
    },
    {
        'lower': 0.0,
        'upper': 200000.0
    },
    {
        'lower': 0.0,
        'upper': 150000.0
    },
    {
        'lower': 0.0,
        'upper': 2500.0
    },
    {
        'lower': 0.0,
        'upper': 20.0
    },
    {
        'lower': -2700000.0,
        'upper': 2700000.0
    }
]

POPSIZE = 50
MAXGENS = 50
NVARS = 3
PXOVER = 0.8
PMUTATION = 0.15

# gene = {
#     lower:
#     upper:
#     value:
# }
# population = [
#     {
#     genes: array of genes;
#   fitness:
#   rfitness:
#   cfitness:
# }
# ]  size is POSIZE+1

gene = dict()
population = list()
new_population = list()

complete_report = list()


# for utility only
def copy(data):
    return pickle.loads(pickle.dumps(data))


# initialize population with random value of genes
def initialize():
    for i in range(POPSIZE):
        population.append({
            'fitness': 0,
            'rfitness': 0,
            'cfitness': 0,
            'genes': [{'lower': item['lower'], 'upper':item['upper'], 'value':np.random.uniform(item['lower'], item['upper'])} for item in bounds]
        })


# evaluate function x^2-xy+yz where x,y,z are genes
def evaluate():
    for i in range(POPSIZE):
        a1 = population[i]['genes'][0]['value']
        a2 = population[i]['genes'][1]['value']
        a3 = population[i]['genes'][2]['value']
        a4 = population[i]['genes'][3]['value']
        a5 = population[i]['genes'][4]['value']
        b = population[i]['genes'][5]['value']
        y_hat = a1 * X_train['Avg. Area Income'] + a2 * X_train['Avg. Area House Age'] + a3 * \
            X_train['Avg. Area Number of Rooms'] + a4 * \
            X_train['Avg. Area Number of Bedrooms'] + \
            a5 * X_train['Area Population'] + b
        rmse = np.sqrt(metrics.mean_squared_error(y_hat, y_train))
        population[i]['fitness'] = rmse


# for crossover operation
def Xover(one, two):
    point = np.random.randint(0, NVARS-1)
    for i in range(point):
        t = copy(population[one]['genes'][i])
        population[one]['genes'][i] = copy(population[two]['genes'][i])
        population[two]['genes'][i] = copy(t)


# variable 'first' in below code keep track of individual choosen for crossover
def crossover():
    a = 0.0
    b = 1.0
    first = 0
    one = 0
    for mem in range(POPSIZE):
        x = np.random.uniform(a, b)
        if x < PXOVER:
            first += 1
            if first % 2 == 0:
                Xover(one, mem)
            else:
                one = mem


# updating last item of population with highest fitness value genotype i.e. individual
def keep_the_best():
    total_population = len(population)
    max_fit_gene_index = np.argmin(
        [population[i]['fitness'] for i in range(total_population)])
    if total_population == POPSIZE:
        population.append(copy(population[max_fit_gene_index]))
    else:
        population[POPSIZE] = copy(population[max_fit_gene_index])


# mutate genes with random value with in bounds if x generated randomly is less than mutaion probability
def mutate():
    for i in range(POPSIZE):
        a = 0.0
        b = 1.0
        x = np.random.uniform(a, b)
        if x < PMUTATION:
            population[i]['genes'] = copy([{'lower': gene['lower'], 'upper':gene['upper'], 'value':np.random.uniform(
                gene['lower'], gene['upper'])} for gene in population[i]['genes']])


# select new generation using proportional selection operator
def selector():
    a = 0.0
    b = 1.0
    for i in range(POPSIZE):
        random_index = np.random.randint(0, POPSIZE)
        if len(new_population) != POPSIZE:
            new_population.append(copy(population[i]))
        else:
            new_population[i] = copy(population[i])

    population[:POPSIZE] = copy(new_population)


# Calculate report for each generation
def report(generation):
    fitness_values = [population[i]['fitness'] for i in range(POPSIZE)]
    return {
        'Generation Number': generation,
        'Best Value': population[np.argmin([population[i]['fitness'] for i in range(POPSIZE)])]['fitness'],
        'Average Fitness': np.average(fitness_values),
        'Standard Deviation': np.std(fitness_values)
    }


def elitist():
    current_best_individual_index = np.argmin(
        [population[i]['fitness'] for i in range(POPSIZE)])
    current_worst_individual_index = np.argmax(
        [population[i]['fitness'] for i in range(POPSIZE)])
    previous_best_individual_index = POPSIZE
    if population[current_best_individual_index]['fitness'] <= population[POPSIZE]['fitness']:
        population[POPSIZE] = copy(population[current_best_individual_index])
    else:
        population[current_worst_individual_index] = copy(population[POPSIZE])


if __name__ == '__main__':
    USAhousing = pd.read_csv('USA_Housing.csv')
    X = USAhousing[['Avg. Area Income', 'Avg. Area House Age',
                    'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
    y = USAhousing['Price']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=101)
    initialize()
    evaluate()
    keep_the_best()
    for i in range(MAXGENS):
        selector()
        crossover()
        mutate()
        complete_report.append(report(i))
        evaluate()
        elitist()
    print(population[POPSIZE]['genes'])
    print('Bounds are ', str(bounds))
    print('Best Finess Value ', complete_report[-1]['Best Value'])
    a1, a2, a3, a4, a5, b = [gene['value']
                             for gene in population[POPSIZE]['genes']]
    predictions = a1 * X_test['Avg. Area Income'] + a2 * X_test['Avg. Area House Age'] + a3 * \
        X_test['Avg. Area Number of Rooms'] + a4 * \
        X_test['Avg. Area Number of Bedrooms'] + \
        a5 * X_test['Area Population'] + b
    plt.scatter(y_test, predictions)
    plt.show()
