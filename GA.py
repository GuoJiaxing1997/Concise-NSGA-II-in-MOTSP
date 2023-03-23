import random

# GA的流程，其中交叉函数确保了不会产生非法解（遗漏城市或一个城市遍历了两次）
# GA process, in which the crossover function ensures that no illegal solution will be generated (missing a city or traversing a city twice)
class GA(object):

    def __init__(self, parent_population, population_size, length_chromosome):

        self.parent_population = parent_population
        self.population_size = population_size
        self.length_chromosome = length_chromosome

        self.offspring = []

        self.parentA = [None] * length_chromosome
        self.parentB = [None] * length_chromosome
        self.childA = [None] * length_chromosome
        self.childB = [None] * length_chromosome

        # 选择，交叉，变异的决策变量。用来调整函数效果以及基础算法后的优化用，此代码未使用
        # decision variables of selection, crossover, and mutation. Used to adjust the function effect and optimize. This example is not used
        self.selection_variable = 1
        self.crossover_variable = 1
        self.mutation_variable = 1

    def Run(self):
        while len(self.offspring) != len(self.parent_population):

            # 选择，选了两个个体，赋值给了self.parentA，self.parentAB
            # select two individuals, and assign to self.parentA, self.parentB
            self.Selection(self.parent_population)
            # 交叉，赋值给了self.childA, self.childB
            # Cross, assigned to self.childA, self.childB
            self.Crossover(self.parentA, self.parentB)

            self.Mutation(self.childA)
            self.Mutation(self.childB)

            self.offspring.append(self.childA[:])
            self.offspring.append(self.childB[:])


        print('Qt by GA():', self.offspring)
        return self.offspring

    def Selection(self, population):
        a, b = random.sample(range(self.population_size), 2)
        self.parentA = population[a][:]
        self.parentB = population[b][:]

    def Crossover(self, parentA, parentB):
        # 随机选择交叉区间的起始和结束位置
        # Randomly select the starting and ending positions of the crossover interval
        crossover_start = random.randint(0, self.length_chromosome - 2)
        crossover_end = random.randint(crossover_start + 1, self.length_chromosome - 1)

        # 获取子代的交叉区间
        # Obtain the crossover interval of the offspring
        childA_segment = parentA[crossover_start:crossover_end + 1]
        childB_segment = parentB[crossover_start:crossover_end + 1]

        childA = [None] * self.length_chromosome
        childB = [None] * self.length_chromosome

        # 将交叉区间插入子代基因
        # Inserting crossover sections into offspring genes
        childA[crossover_start:crossover_end + 1] = childA_segment
        childB[crossover_start:crossover_end + 1] = childB_segment

        # 从parentB中获取不在交叉区间内的城市，按照parentB的顺序填充childA
        # Obtain cities that are not within the crossover sections from parentB, and fill in childA in the order of parentB
        idxA = 0
        for city in parentB:
            if city not in childA_segment:
                while childA[idxA] is not None:
                    idxA += 1
                childA[idxA] = city

        # 从parentA中获取不在交叉区间内的城市，按照parentA的顺序填充childB
        # Obtain cities that are not within the crossover sections from parentA, and fill in childB in the order of parentA
        idxB = 0
        for city in parentA:
            if city not in childB_segment:
                while childB[idxB] is not None:
                    idxB += 1
                childB[idxB] = city

        self.childA = childA[:]
        self.childB = childB[:]

    def Mutation(self, life):
        # 产生两个随机点
        # Generate 2 Random Points
        a, b = sorted(random.sample(range(self.length_chromosome), 2))
        life[a], life[b] = life[b], life[a]



