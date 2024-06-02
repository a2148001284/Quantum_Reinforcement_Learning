import matplotlib.pyplot as plt
import numpy as np

file_path1="q_cost_ddqn.txt"
list1 = []
#list2 = []
with open(file_path1,"r") as f:
    for line in f:
        items = line.strip().split()
        list1.append(float(items[0]))

file_path2="q_steps_ddqn.txt"
list3 = []
with open(file_path2,"r") as f:
    for line in f:
        items = line.strip().split()
        list3.append(int(items[0]))

# 绘制损失图
def plot_cost():
    # np.arange(3)产生0-2数组
    #plt.plot(np.arange(len(self.cost)), self.cost)
    plt.plot(np.arange(len(list1)),list1)
    plt.xlabel("step")
    plt.ylabel("cost")
    plt.show()


# 绘制每轮需要走几步
def plot_steps_of_each_episode():
    #plt.plot(np.arange(len(self.steps_of_each_episode)), self.steps_of_each_episode)
    plt.plot(np.arange(len(list3)), list3)
    plt.xlabel("episode")
    plt.ylabel("done steps")
    plt.show()

plot_cost()
plot_steps_of_each_episode()