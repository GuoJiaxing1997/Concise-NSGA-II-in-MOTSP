# Concise-NSGA-II-in-MOTSP

一个简明扼要的NSGA-II算法实例，用于解决多目标TSP（MOTSP），适用于初学者与大学生

a Concise NSGA-II algorithm example in Multi-objective TSP (MOTSP) for beginners and College Students

提供丰富的中英文注释，可帮你快速理解整个算法流程

Rich Chinese and English Notes are provided to help you quickly understand the entire algorithm process, with the help of a translator, please understand

如果你不知道从何下手学习，强烈安利此项目！

If you don't know where to start, this project is highly recommended!

菜鸡一枚，已尽可能注释。如有疏漏与错误，请包含

Limited personal level. Please forgive any omissions or errors

作者：吴，某B大学，中国

author: Wu, N?E? University, China

## NSGA2.py

三个函数：快速非支配排序，拥挤度计算，基于拥挤度的排序，为此类的主要功能。代码中提供了丰富的注释，在此不过多赘述。

Three functions: FastNonDominatedSort(), CrowdingDistance(), CrowdingDistanceSort() are the main functions of this class. The code provides a wealth of Notes, no more detail here.

考虑了两个目标函数：TSP的总距离与总成本

Two objective functions are considered: total distance and total cost of TSP

## GenerateOffspring.py

GA的流程，其中交叉函数确保了不会产生非法解（遗漏城市或一个城市遍历了两次）

GA process, in which the crossover function ensures that no illegal solution will be generated (missing a city or traversing a city twice)

## Data.py and Decode.py

输入了一个距离矩阵和成本矩阵。在Decode中，可将当前种群的两个目标函数列表计算并返回。

A distance matrix and a cost matrix have been entered. In Decode, you can calculate and return a list of two objective functions for the current population.
