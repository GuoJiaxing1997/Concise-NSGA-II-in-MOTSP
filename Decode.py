from Data import Data

class Decode(object):

    def __init__(self, population):
        
        self.population = population
        # 1对应distance，2对应cost
        # v1 is distance, v2 is cost
        self.obj_func_v1 = []
        self.obj_func_v2 = []

    def CaculateObjFuncV1(self):
        for i in range(len(self.population)):
            value = 0
            for j in range(len(self.population[0]) - 1):
                # 注意：应该是[self.population[i][j + 1] - 1]，而不是[self.population[i][j + 1]]
                # NOTE: Should be [self.population[i][j + 1] - 1] instead of [self.population[i][j + 1]]
                value += Data().distance_matrix[self.population[i][j] - 1][self.population[i][j + 1] - 1]
            value += Data().distance_matrix[self.population[i][-1] - 1][self.population[i][0] - 1]
            self.obj_func_v1.append(value)
        return self.obj_func_v1[:]

    def CaculateObjFuncV2(self):
        for i in range(len(self.population)):
            value = 0
            for j in range(len(self.population[0]) - 1):
                value += Data().cost_matrix[self.population[i][j] - 1][self.population[i][j + 1] - 1]
            value += Data().cost_matrix[self.population[i][-1] - 1][self.population[i][0] - 1]
            self.obj_func_v2.append(value)
        return self.obj_func_v2[:]

