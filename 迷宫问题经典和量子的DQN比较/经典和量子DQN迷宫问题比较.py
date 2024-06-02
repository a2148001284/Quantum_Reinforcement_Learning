from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import rl_utils

file_path1="jingdian_steps.txt"
file_path2="q_steps.txt"
list1 = []
list2 = []
list3 = []
list4 = []
with open(file_path1,"r") as f:
    for line in f:
        items = line.strip().split()
        if int(items[0])>672:
            break
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

# def drawpic1():
#     plt.plot(list1, list2)
#     plt.xlabel('Epsiode')
#     plt.ylabel('Collected rewards')
#     #plt.legend()
#     plt.show()
#
# def drawpic2():
#     plt.plot(list3, list4)
#     plt.xlabel('Epsiode')
#     plt.ylabel('Collected rewards')
#     plt.show()

#画组合的图
def opt2():
    plt.plot(list1,list2,label="DQN",color="orange")
    plt.plot(list3,list4,label="Q-DQN",color="blue")
    plt.xlabel('Epsiode')
    plt.ylabel('Steps')
    plt.legend()
    plt.show()


# drawpic1()
# drawpic2()
opt2()