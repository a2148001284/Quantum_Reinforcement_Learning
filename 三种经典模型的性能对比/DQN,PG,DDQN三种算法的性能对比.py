from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import rl_utils

file_path1="DDQN.txt"
file_path2="DQN.txt"
file_path3="PG.txt"
list1 = []
list2 = []
list3 = []
list4 = []
list5 = []
list6 = []
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
        if int(items[0].split('.')[0])>200:
            break
        list5.append(int(items[0].split('.')[0]))
        list6.append(int(items[1].split('.')[0]))
#展示增速图部分
list2=rl_utils.moving_average(list2, 9)
list4=rl_utils.moving_average(list4, 9)
list6=rl_utils.moving_average(list6, 9)

#画组合的图
def opt2():
    plt.plot(list1,list2,label="DQQN",color="orange")
    plt.plot(list3,list4,label="DQN",color="blue")
    plt.plot(list5, list6, label="PG", color="green")
    plt.xlabel('Epsiode')
    plt.ylabel('Collected rewards')
    plt.legend()
    plt.show()


opt2()