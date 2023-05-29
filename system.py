import numpy as np
import pandas as pd


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

    def __init__(self, nrofcities) -> None:
        self.nrofcities = nrofcities
        self.starting_city = np.random.randint(1, nrofcities)
        self.route_len = nrofcities - 1
        self.current_city = self.starting_city
        self.reset_avaliable_targets()

        self.visited = []
        self.total_distance = 0.0
        self.decisions = None
        self.route_probabilities = None
        self.route_distance = 0.0

    def make_decision_table(self):
        likelyhood = {}
        for target_city in self.avaliable_targets:
            dist = (1/Ant.distgraph[self.current_city]
                    [target_city]) ** Ant.beta
            pher = Ant.pherograph[self.current_city][target_city] ** Ant.alpha
            likelyhood[target_city] = dist * pher

        likely_sum = sum([i for i in likelyhood])

        self.decisions = {target_city : likelyhood[target_city] /
                          likely_sum for target_city in self.avaliable_targets}

    def make_route_probability(self):
        sum_decisions = sum(self.decisions)
        self.route_probabilities = {target : i/sum_decisions for target,i in zip(self.avaliable_targets, self.decisions)}

    def roulette(self):
        randomval = np.random.uniform()

        for i, p in self.route_probabilities.items():
            randomval -= p
            if randomval <= 0:
                self.current_city = i
                self.avaliable_targets.remove(self.current_city)
                self.visited.append(self.current_city)
                break

    def reset_avaliable_targets(self):
        self.avaliable_targets = [i+1 for i in range(self.nrofcities)]
        self.avaliable_targets.remove(self.current_city)

    def calculate_route_length(self): 
        self.route_distance = Ant.distgraph[self.visited[-1]][self.visited[0]]
        for loc, dest in zip(self.visited, self.visited[1:]):
            self.route_distance += Ant.distgraph[loc][dest]
        raise NotImplementedError

    def travel(self):
        self.current_city = self.starting_city
        self.visited = [self.starting_city]
        self.reset_avaliable_targets()

        for _ in range(self.route_len):
            self.make_decision_table()
            self.make_route_probability()
            self.roulette()
        # self.visited.append(self.starting_city) # I think it may be harmful to do that


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

    def evaporate_pheromone(self):
        self.distgraph = self.distgraph * self.ro

    def deposit_pheromones(self):
        for ant in self.population:
            deposit = 1/ant.route_distance
            self.pherograph[ant.visited[-1]][ant.visited[0]] += deposit
            for origin, destination in zip(ant.visited, ant.visited[1:]):
                self.pherograph[origin][destination] += deposit

        Ant.pherograph = self.pherograph

    def define_best_ant(self) -> Ant:
        distances = pd.Series([ant.route_distance for ant in self.population])
        return self.population[distances.idxmin()]

    def run(self):
        for _ in range(self.maxiter):
            for ant in self.population:
                ant.travel()
            self.evaporate_pheromone()
            self.deposit_pheromones()

        print(self.define_best_ant())


game = System(
    "Traveling Salesman Problem Data-20230314\\cities_4.txt", 100, 0.5)

game.run()