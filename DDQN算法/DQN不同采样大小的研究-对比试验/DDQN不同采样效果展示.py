from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import rl_utils

file_path1="DQN不同采样大小的研究 采样16 阈值200.txt"
file_path2="DQN不同采样大小的研究 采样32 阈值200.txt"
file_path3="DQN不同采样大小的研究 采样64 阈值200.txt"
file_path4="DQN不同采样大小的研究 采样128 阈值200.txt"
list1 = []
list2 = []
list3 = []
list4 = []
list5 = []
list6 = []
list7 = []
list8 = []
with open(file_path1,"r") as f:
    for line in f:
        items = line.strip().split()
        list1.append(int(items[0].split('.')[0]))
        list2.append(int(items[1].split('.')[0]))
with open(file_path2,"r") as f:
    for line in f:
        items = line.strip().split()
        list3.append(int(items[0].split('.')[0]))
        list4.append(int(items[1].split('.')[0]))
with open(file_path3,"r") as f:
    for line in f:
        items = line.strip().split()
        list5.append(int(items[0].split('.')[0]))
        list6.append(int(items[1].split('.')[0]))
with open(file_path4,"r") as f:
    for line in f:
        items = line.strip().split()
        list7.append(int(items[0].split('.')[0]))
        list8.append(int(items[1].split('.')[0]))
#展示增速图部分
# list2=rl_utils.moving_average(list2, 9)
# list4=rl_utils.moving_average(list4, 9)
# list6=rl_utils.moving_average(list6, 9)
# list8=rl_utils.moving_average(list8, 9)

def draw():
    plt.plot(list1,list2,label="DQN-batch16",color="orange")
    plt.plot(list1,list4,label="DQN-batch32",color="blue")
    plt.plot(list1,list6, label="DQN-batch64", color="green")
    plt.plot(list1,list8, label="DQN-batch128", color="red")
    plt.xlabel('Epsiode')
    plt.ylabel('Collected rewards')
    plt.legend()
    plt.show()
def drawpic():
    plt.plot(list7, list8)
    plt.xlabel('Epsiode')
    plt.ylabel('Collected rewards')
    plt.show()
draw()
#drawpic()