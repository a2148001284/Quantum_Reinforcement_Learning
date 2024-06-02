import matplotlib.pyplot as plt

import rl_utils

file_path1="q_cost.txt"
list1 = []
list2 = []
with open(file_path1,"r") as f:
    for line in f:
        items = line.strip().split()#去除了每行数据两端的空白字符（包括换行符等），然后使用 split() 方法将每行数据按空格分割成多个部分，并将这些部分存储在列表 items 中
        #list1.append(int(items[0].split('.')[0])) #这行代码将每行数据的第一个部分（通过索引 0 访问）按小数点分割后的第一个部分转换为整数，并将其添加到 list1 列表中
        #list2.append(float(items[1].split('.')[0]))
        if int(items[0])>672:
            break
        list1.append(int(items[0]))
        list2.append(float(items[1]))

file_path2="q_steps.txt"
list3 = []
list4 = []
with open(file_path2,"r") as f:
    for line in f:
        items = line.strip().split()
        list3.append(int(items[0].split('.')[0]))
        list4.append(int(items[1].split('.')[0]))

list4=rl_utils.moving_average(list4, 9)

# 绘制损失图
def plot_cost():
    # np.arange(3)产生0-2数组
    #plt.plot(np.arange(len(self.cost)), self.cost)
    plt.plot(list1,list2)
    plt.xlabel("step")
    plt.ylabel("cost")
    plt.show()


# 绘制每轮需要走几步
def plot_steps_of_each_episode():
    #plt.plot(np.arange(len(self.steps_of_each_episode)), self.steps_of_each_episode)
    plt.plot(list3, list4)
    plt.xlabel("episode")
    plt.ylabel("done steps")
    plt.show()

plot_cost()
plot_steps_of_each_episode()