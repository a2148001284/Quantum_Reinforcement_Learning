from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import rl_utils

file_path1="DDQN采样池控制16.txt"
file_path2="q-DDQN采样池控制16.txt"
list1 = []
list2 = []
list3 = []
list4 = []
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
#展示增速图部分
# list2=rl_utils.moving_average(list2, 9)
# list4=rl_utils.moving_average(list4, 9)

#画DDQN的效果图
def drawpic1():
    plt.plot(list1, list2)
    #plt.xticks(range(0,3000,500))  #range()三个参数分别是起始值，结束值以及步长(可选) 注意：不包括终止值
    #plt.yticks(range(0,201,50))
    #plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(100))
    #plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.xlabel('Epsiode')
    plt.ylabel('Collected rewards')
    #plt.legend()
    plt.show()

#画q-DDQN的效果图
def drawpic2():
    plt.plot(list3, list4)
    plt.xlabel('Epsiode')
    plt.ylabel('Collected rewards')
    plt.show()

#画两者之间的比较效果图
def drawpic3():
    plt.plot(list1,list2,label="DDQN",color="orange")
    plt.plot(list3,list4,label="Q-DDQN",color="blue")
    plt.xlabel('Epsiode')
    plt.ylabel('Collected rewards')
    plt.legend()
    plt.show()

#drawpic1()
#drawpic2()
drawpic3()