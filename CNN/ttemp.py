import argparse
# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
parser.add_argument('-weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('-freeze_GRU', type=int, default=0, help='freeze GRU layer or not')
parser.add_argument('-threathhold', type=float, default=0.4, help='threathhold')
parser.add_argument('-hidden_size2', type=int, default=128, help='hidden size 2: 128->Hidden_size2->Hidden_size3->1')
parser.add_argument('-hidden_size3', type=int, default=64, help='hidden size 3: 128->Hidden_size2->Hidden_size3->1')
parser.add_argument('-epoch', type=int, default=100, help='number of epochs')

# 设置随机种子
# seed = 3407
# torch.manual_seed(seed)

# 超参数
# 解析参数
args = parser.parse_args()
# =============================================
# 是否冻结 GRU 层的所有权重: 1冻结，0不冻结
freeze_GRU = args.freeze_GRU # 冻结的效果更好
lr = args.lr
weight_decay = args.weight_decay
epochs = args.epoch
hidden_size2 = args.hidden_size2
hidden_size3 = args.hidden_size3
threathhold = args.threathhold

# 把它们打印出来看看
print(f'freeze_GRU: {freeze_GRU}')
print(f'lr: {lr}')
print(f'weight_decay: {weight_decay}')
print(f'epochs: {epochs}')
print(f'hidden_size2: {hidden_size2}')
print(f'hidden_size3: {hidden_size3}')
print(f'threathhold: {threathhold}')
# =============================================


from deap import base, creator, tools, algorithms
import random
import subprocess

# 定义适应度函数
def evalOneMax(individual):
    command = f"/Data4/gly_wkdir/environment/DeepLpy3.9/bin/python /Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/CNN/ttemp.py -lr {individual[0]} -weight_decay {individual[1]} -freeze_GRU {individual[2]} -threathhold {individual[3]} -hidden_size2 {individual[4]} -hidden_size3 {individual[5]} -epoch 100"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    loss = float(result.stdout.split()[-1])  # 假设最后打印的是损失值
    return (1 / (1 + loss),)  # 适应度函数，适应度越高越好

# 定义个体和种群
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0.001, 0.01)  # 学习率
toolbox.register("attr_weight_decay", random.uniform, 0.0001, 0.001)  # 权重衰减
toolbox.register("attr_freeze_GRU", random.randint, 0, 1)  # 是否冻结GRU层
toolbox.register("attr_threathhold", random.uniform, 0.3, 0.5)  # 阈值
toolbox.register("attr_hidden_size2", random.choice, [128, 256])  # hidden_size2
toolbox.register("attr_hidden_size3", random.choice, [64, 128])  # hidden_size3

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_float, toolbox.attr_weight_decay, toolbox.attr_freeze_GRU, toolbox.attr_threathhold, toolbox.attr_hidden_size2, toolbox.attr_hidden_size3), n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# 初始化种群
population = toolbox.population(n=50)

# 进行遗传算法优化
NGEN=40
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

# 打印最优参数
best_ind = tools.selBest(population, 1)[0]
print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))