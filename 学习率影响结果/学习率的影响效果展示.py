from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import rl_utils

#用于展示实验4
file_path1="learn_0.2.txt"
file_path2="learn_0.1.txt"
file_path3="learn_0.01.txt"
file_path4="learn_0.001.txt"
file_path5="learn_0.0001.txt"
list1 = []
list2 = []

list3 = []
list4 = []

list5 = []
list6 = []

list7 = []
list8 = []

list9 = []
list10 = []
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
with open(file_path5,"r") as f:
    for line in f:
        items = line.strip().split()
        list9.append(int(items[0].split('.')[0]))
        list10.append(int(items[1].split('.')[0]))

list2=rl_utils.moving_average(list2, 9)
list4=rl_utils.moving_average(list4, 9)
list6=rl_utils.moving_average(list6, 9)
list8=rl_utils.moving_average(list8, 9)
list10=rl_utils.moving_average(list10, 9)

plt.plot(list1,list2,label="Learn-ratio:0.2",color="green")
plt.plot(list3,list4,label="Learn-ratio:0.1",color="orange")
plt.plot(list5,list6,label="Learn-ratio:0.01",color="blue")
plt.plot(list7,list8,label="Learn-ratio:0.001",color="red")
plt.plot(list9,list10,label="Learn-ratio:0.0001",color="black")
plt.xlabel('Epsiode')
plt.ylabel('Collected rewards')
plt.legend()
plt.show()
