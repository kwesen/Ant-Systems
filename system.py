import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

def euclidean_dist(city1: tuple, city2: tuple) -> float:
    x1, y1 = city1
    x2, y2 = city2
    dist = np.sqrt(np.power(x1-x2, 2) + np.power(y1-y2, 2))
    return dist


def city_dist_map(cities: list[tuple]) -> tuple[pd.DataFrame]:
    distance_map = {i+1: {} for i in range(len(cities))}
    for key, city in enumerate(cities):
        for destkey, destination in enumerate(cities):
            distance_map[key+1][destkey+1] = euclidean_dist(city, destination)
    distance_map = pd.DataFrame(distance_map)
    pheromoneinit = 1/distance_map.values.max()
    pheromone_map = {i+1: {} for i in range(len(cities))}
    for key, _ in enumerate(cities):
        for destkey, _ in enumerate(cities):
            pheromone_map[key+1][destkey+1] = pheromoneinit
    pheromone_map = pd.DataFrame(pheromone_map)

    return distance_map, pheromone_map


def city_cord_reader(path: str) -> list[tuple[float, float]]:
    '''Reads the file provided as a path that contains the coordinates of cities\n
    The coordinates should be written in seperate lines in [ ] square
    brackets and separated by spaces'''

    with open(path, 'r+') as file:
        usefull_data = file.readlines()[1:]
        ix1, ix2 = usefull_data[0].find('[')+1, usefull_data[0].find(']')
        iy1, iy2 = usefull_data[1].find('[')+1, usefull_data[1].find(']')
        x, y = usefull_data[0][ix1:ix2], usefull_data[1][iy1:iy2]
        x, y = x.split(' '), y.split(' ')
        x = [i for i in x if i]
        y = [i for i in y if i]
        if len(x) != len(y):
            raise Exception("Lengths don't match, file may be corrupted")
        cords = []
        for xval, yval in zip(x, y):
            cords.append((float(xval), float(yval)))
    return cords


class Ant:
    distgraph = None
    pherograph = None

    alpha = 1
    beta = 5

    def __init__(self, nrofcities: int) -> None:
        self.nrofcities = nrofcities
        self.starting_city = np.random.randint(1, nrofcities+1)
        self.route_len = nrofcities - 1
        self.current_city = self.starting_city
        self.visited = [self.starting_city]

        self.reset_avaliable_targets()

        self.decisions = None
        self.route_probabilities = None
        self.route_distance = 0.0

    def make_decision_table(self):
        likelyhood = {}
        for target_city in self.available_targets:
            dist = (1/Ant.distgraph[self.current_city]
                    [target_city]) ** Ant.beta
            pher = Ant.pherograph[self.current_city][target_city] ** Ant.alpha
            likelyhood[target_city] = dist * pher

        likely_sum = sum([val for _, val in likelyhood.items()])

        self.decisions = {target_city : likelyhood[target_city] /
                          likely_sum for target_city in self.available_targets}

    def make_route_probability(self):
        sum_decisions = sum([val for _, val in self.decisions.items()])
        self.route_probabilities = {target : self.decisions[target] /
                                    sum_decisions for target in self.available_targets}

    def roulette(self):
        randomval = np.random.uniform(0,1)

        for i, p in self.route_probabilities.items():
            randomval -= p
            if randomval <= 0:
                if i == self.current_city:
                    raise Exception("Traveling to the same city")
                if i in self.visited:
                    raise Exception("Traveling to a visited city")
                self.current_city = i
                self.visited.append(self.current_city)
                self.reset_avaliable_targets()
                break

    def reset_avaliable_targets(self):
        self.available_targets = [i+1 for i in range(self.nrofcities) if i+1 not in self.visited]

    def calculate_route_length(self): 
        self.route_distance = Ant.distgraph[self.visited[-1]][self.visited[0]]
        for loc, dest in zip(self.visited, self.visited[1:]):
            self.route_distance += Ant.distgraph[loc][dest]

    def travel(self):
        self.current_city = self.starting_city
        self.visited = [self.starting_city]
        self.reset_avaliable_targets()

        for _ in range(self.route_len):
            self.make_decision_table()
            self.make_route_probability()
            self.roulette()
        self.calculate_route_length()


class System:
    def __init__(self, path: str, maxiter: int, ro: float) -> None:
        self.maxiter = maxiter
        self.ro = ro

        self.cities = city_cord_reader(path)
        self.distgraph, self.pherograph = city_dist_map(self.cities)
        Ant.distgraph = self.distgraph
        Ant.pherograph = self.pherograph

        self.population = [Ant(len(self.cities))
                           for _ in range(len(self.cities))]
        self.Ants_all: list[Ant] = []

    def evaporate_pheromone(self):
        self.pherograph = self.pherograph * (1 - self.ro)

    def deposit_pheromones(self):
        for ant in self.population:
            deposit = 1/ant.route_distance
            self.pherograph[ant.visited[-1]][ant.visited[0]] += deposit
            self.pherograph[ant.visited[0]][ant.visited[-1]] += deposit
            for origin, destination in zip(ant.visited, ant.visited[1:]):
                self.pherograph[origin][destination] += deposit
                self.pherograph[destination][origin] += deposit

        Ant.pherograph = self.pherograph

    def define_best_ant(self) -> Ant:
        distances = pd.Series([ant.route_distance for ant in self.Ants_all])
        return self.Ants_all[distances.idxmin()]

    def run(self):
        for _ in range(self.maxiter):
            for ant in self.population:
                ant.travel()
            self.evaporate_pheromone()
            self.deposit_pheromones()
            self.Ants_all.extend(deepcopy(self.population))
        self.plot_path(self.define_best_ant())


    def plot_path(self, ant: Ant):
        plt.figure()
        for loc in self.cities:
            for dest in self.cities:
                x1, y1 = loc
                x2, y2 = dest
                plt.plot((x1, x2),(y1, y2),'b--', alpha = 0.1)

        for loc, dest in zip(ant.visited, ant.visited[1:]):
            x1, y1 = self.cities[loc-1]
            x2, y2 = self.cities[dest-1]
            plt.plot((x1, x2),(y1, y2),'r')
        print(ant.visited)
        print(ant.route_distance)
        x1, y1 = self.cities[ant.visited[-1]-1]
        x2, y2 = self.cities[ant.visited[0]-1]
        plt.plot((x1, x2),(y1, y2),'r')
        for i, city in enumerate(self.cities):
            x, y = city
            plt.plot(x, y, 'bo')
            plt.text(x+0.1,y+0.1,str(i+1),color='#000000')
        plt.title(f"Distance = {ant.route_distance : .05f}")
        plt.axis("off")
        plt.show()



game = System(
    "Traveling Salesman Problem Data-20230314\cities_4.txt", 100, 0.5)

game.run()