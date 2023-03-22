# -*- coding: utf-8 -*-
# @Author : Wu, N?E? University, China
# @GitHub : GuoJiaxing1997
# @Time : 2023/3/22 15:59
# 英文与中文注释
# English and Chinese Notes, with the help of a translator, please understand
# 水平有限，已尽可能注释。如有疏漏与错误，请包含
# Limited personal level, notes have been provided as much as possible. Please forgive any omissions or errors

import random
import matplotlib.pyplot as plt
from Data import Data
from Decode import Decode
from GA import GA


class NSGA2(object):

    def __init__(self):
        self.population_size = 30
        self.length_chromosome = Data().length_chromosome
        self.population = []
        # 两个目标函数，分别为距离与成本
        # Two objective functions, namely distance and cost
        self.obj_func_v1 = []
        self.obj_func_v2 = []
        # 各等级的帕累托前沿
        # Pareto Fronts at Each Level
        self.fronts = [[]]
        # 迭代次数
        # Total Iterations
        self.num_iterations = 500
        # 迭代计数
        # Iteration counter
        self.iteration_counter = 0

    def InitializePopulation(self, population_size, length_chromosome):
        # 随机初始化一种群
        # Initialize a population randomly
        permutations = []
        for i in range(population_size):
            tour = [j + 1 for j in range(length_chromosome)]
            random.seed()
            random.shuffle(tour)
            permutations.append(tour[:])
            pass
        return permutations[:]

    def FastNonDominatedSort(self, obj_func_v1, obj_func_v2):
        # 注意：在进行快速非支配排序时，此时Pt+Qt中的个体数量是population_size的二倍！
        # 为方便理解，“p支配q”在下文中将口语化为“p比q强”
        # Note: When FastNonDominatedSort, the number of individuals in Pt+Qt is population_size * 2!
        # To facilitate understanding, "p dominates q" is colloquially defined as "p is stronger than q" in the following text

        fronts = [[]]
        # 比p弱的个体组成的集合
        # A collection of individuals weaker than p
        S = [[] for i in range(self.population_size * 2)]
        # 比p强的个体的个数
        # Number of individuals who are stronger than p
        n = [0 for i in range(self.population_size * 2)]

        rank = [0 for i in range(self.population_size * 2)]
        for p in range(0, self.population_size * 2):
            S[p] = []
            n[p] = 0

            # 此循环的目的，是为了找到最好的Pareto前沿，也称为F1，即fronts[0]
            # The purpose of this for() is to find the best Pareto frontier, also known as F1, i.e. fronts[0]
            for q in range(0, self.population_size * 2):
                # 对每一个个体p，找出其支配的个体，存到S中。找到支配p的个体，将其数量存在n中
                # For each individual p, find out the weaker and store it in S. Find out the stronger and store its number in n
                if (obj_func_v1[p] < obj_func_v1[q] and obj_func_v2[p] <
                        obj_func_v2[q]) or (
                        obj_func_v1[p] <= obj_func_v1[q] and obj_func_v2[p] <
                        obj_func_v2[q]) or (
                        obj_func_v1[p] < obj_func_v1[q] and obj_func_v2[p] <=
                        obj_func_v2[q]):
                    # 如果q不在比p弱的列表中，则将q加入
                    # If q is not in a weaker list than p, add q to the list
                    if q not in S[p]:
                        S[p].append(q)
                elif (obj_func_v1[q] < obj_func_v1[p] and obj_func_v2[q] <
                        obj_func_v2[p]) or (
                        obj_func_v1[q] <= obj_func_v1[p] and obj_func_v2[q] <
                        obj_func_v2[p]) or (
                        obj_func_v1[q] < obj_func_v1[p] and obj_func_v2[q] <=
                        obj_func_v2[p]):
                    # 否则q比p强，计数+1
                    # Otherwise, q is stronger than p, count+1
                    n[p] = n[p] + 1
            # 如果n[p] = 0，说明p是一个非支配解，将它的rank的级别设为最低
            # If n [p]=0, it indicates that p is a nondominant solution, and its rank level is set to the lowest
            if n[p] == 0:
                rank[p] = 0
                # 同时将p加入到Pareto前沿中
                # add p to the Pareto front
                if p not in fronts[0]:
                    # 这里存放的都是当前最好的个体
                    # These are all the best individuals currently stored here
                    fronts[0].append(p)

        i = 0
        # 此时，已有最好的解集F1，继续求得F2, F3, F4……
        # At this point, there is the best solution set F1, and continue to find F2, F3, F4
        while (fronts[i] != []): # 整个种群是否已经全部分级 Whether the entire population has been fully ranked
            # Q存放后续的非支配解
            # Q Store subsequent nondominated solutions
            Q = []
            for p in fronts[i]:
                # SID:1E8484
                ################################################
                # i = 0时，对于当前p对应的弱者集合S[p]中的每个q，若n[q] = n[q] - 1后n[q] == 0，表明原先n[q] = 1
                # 此时表明，q属于仅次于F1（F1的n[q] = 0）的解集F2（F2的n[q] = 1），并将q的等级赋值为0+1，以此类推，即可对整个种群中的每个个体分级
                # When i = 0, for each q in the weak set S[p] corresponding to the current p, if n[q] == 0 after n[q] = n[q] - 1, it indicates that the original n[q]=1
                # At this point, q belongs to the solution set F2 (F2's n[q] = 1) which is only inferior to F1 (F1's n[q] = 0), and q's rank is assigned as 0+1 accordingly.
                # By doing this for all individuals in the population, each individual can be ranked.
                ################################################
                for q in S[p]:
                    n[q] = n[q] - 1
                    if (n[q] == 0):
                        rank[q] = i + 1
                        if q not in Q:
                            # 将已经被分级的q加入到Q中
                            # Add the ranked q to Q
                            Q.append(q)
            i = i + 1
            # 此时的Q为某一个等级对应的Fi，将其加入fronts
            # At this time, Q is the Fi corresponding to a certain level, add it to fronts
            fronts.append(Q)

        ################################################
        # 最后一层 Pareto 前沿可能会包含一些非常接近甚至等价于其他前沿中的个体，这些个体并没有对优化结果产生任何有意义的贡献。
        # 为了减少冗余计算和输出，删除最后一层 Pareto 前沿是一个常见的做法。
        # The last front may contain individuals that are very close to or even equivalent to other front,
        # and these individuals have not made any meaningful contributions to the optimization results.
        # In order to reduce redundant computation and output, it is a common practice to delete the last front.
        ################################################
        del fronts[len(fronts) - 1]
        return fronts[:]

    def CrowdingDistance(self, obj_func_v1, obj_func_v2, front):
        n_individuals = len(front)
        distance = [0] * n_individuals

        # 将目标函数值与个体组成元组列表
        # Combining the objective function value with individuals to form a tuple list
        sorted_by_obj_func_v1 = [(obj_func_v1[i], front[i]) for i in range(n_individuals)]
        sorted_by_obj_func_v2 = [(obj_func_v2[i], front[i]) for i in range(n_individuals)]

        # 按目标函数值进行排序
        # Sort by objective function value
        sorted_by_obj_func_v1.sort(key=lambda x: x[0])
        sorted_by_obj_func_v2.sort(key=lambda x: x[0])

        # 设置边界点的距离为无穷大
        # Set the distance of the boundary point to infinity
        distance[0] = distance[-1] = float('inf')

        # 计算拥挤度
        # Calculate congestion
        for i in range(1, n_individuals - 1):
            distance[i] += (sorted_by_obj_func_v1[i + 1][0] - sorted_by_obj_func_v1[i - 1][0]) / (
                    sorted_by_obj_func_v1[-1][0] - sorted_by_obj_func_v1[0][0])
            distance[i] += (sorted_by_obj_func_v2[i + 1][0] - sorted_by_obj_func_v2[i - 1][0]) / (
                    sorted_by_obj_func_v2[-1][0] - sorted_by_obj_func_v2[0][0])

        return distance[:]

    def CrowdingDistanceSort(self, obj_func_v1, obj_func_v2, fronts):
        new_population = []
        remaining_space = self.population_size

        for front in fronts:
            if len(front) <= remaining_space:
                for i in front:
                    new_population.append(self.population[i][:])
                remaining_space -= len(front)
            else:
                # 计算当前前沿的拥挤距离
                # Calculate the crowding distance at the current front
                distances = self.CrowdingDistance(obj_func_v1, obj_func_v2, front)
                # 将拥挤距离与front中的个体组成元组列表
                # Combining the crowding distance with the individuals in the front to form a tuple list
                distance_individual_pairs = list(zip(distances, front))
                # 按拥挤距离降序排列
                # Ranked in descending order by crowding distance
                distance_individual_pairs.sort(reverse=True, key=lambda x: x[0])
                for i in range(remaining_space):
                    new_population.append(self.population[distance_individual_pairs[i][1]])
                break
        return new_population[:]

    def Draw(self):
        # 存放坐标点
        # Storage coordinate point
        data_plt = []

        # 绘制第一个前沿，即F1
        # Draw the first front, F1
        for i in range(len(self.fronts[0])):
            data_plt.append([self.obj_func_v1[i], self.obj_func_v2[i]])

        print('Coordinates to draw in Draw()：', data_plt)

        # 创建一个图形对象和一个坐标系对象
        # Create a drawing object and a coordinate system object
        fig, ax = plt.subplots()

        # 遍历数据列表并绘制每个点
        # Traverse the data list and draw each point
        for i, point in enumerate(data_plt):
            # 画每个点，b是蓝色 Draw each point, b is blue
            ax.scatter(point[0], point[1], color='b')
            # 点旁边的序号 Serial number next to the point
            ax.annotate(str(i), (point[0], point[1]), fontsize=8)

        plt.suptitle('The ' + str(self.num_iterations) + 'th Iteration')

        # 设置坐标系尺度
        # Set Coordinate System Scale
        plt.xlim(200, 1000)
        plt.ylim(200, 1000)

        ax.set_xlabel('Total Distance')
        ax.set_ylabel('Total Cost')

        # 显示图形 display graphics
        plt.show()

    def Run(self):
        # 初始化种群（Pt）
        # Initialize Population (Pt)
        self.population = self.InitializePopulation(self.population_size, self.length_chromosome)

        # 打印此fronts中个体的数量
        # Print the number of individuals in this fronts
        def count_fronts_for_print(fronts):
            count = 0
            for row in fronts:
                count += len(row)
            return count

        # 开始迭代
        # Start Iteration
        while self.iteration_counter < self.num_iterations:
            print('****************************************THIS IS THE [' + str(self.iteration_counter + 1) + ']th Iteration****************************************')

            # 将子代Qt加入Pt
            # Add Qt to Pt
            print('Pt:', self.population)
            print('Pt length:', len(self.population))
            self.population.extend(GA(self.population, self.population_size, self.length_chromosome).Run())
            print('Pt+Qt:', self.population)
            print('Pt+Qt length:', len(self.population))

            self.obj_func_v1 = Decode(self.population).CaculateObjFuncV1()
            self.obj_func_v2 = Decode(self.population).CaculateObjFuncV2()
            print('obj_func_v1:', self.obj_func_v1)
            print('obj_func_v1 length:', len(self.obj_func_v1))
            print('obj_func_v2:', self.obj_func_v2)
            print('obj_func_v2 length:', len(self.obj_func_v2))

            print('previous fronts:', self.fronts)
            print('previous fronts length', count_fronts_for_print(self.fronts))
            self.fronts = self.FastNonDominatedSort(self.obj_func_v1, self.obj_func_v2)
            print('fronts:', self.fronts)
            print('fronts length:', count_fronts_for_print(self.fronts))

            self.population = self.CrowdingDistanceSort(self.obj_func_v1, self.obj_func_v2, self.fronts)
            print("Pt+1：", self.population)
            print("Pt+1 length：", len(self.population))

            self.iteration_counter += 1
        # 画图
        # draw
        self.Draw()


if __name__ == '__main__':
    random.seed()
    nsga2 = NSGA2()
    nsga2.Run()
