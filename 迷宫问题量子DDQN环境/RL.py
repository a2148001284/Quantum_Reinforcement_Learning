
import torch
# 用于构建神经网络的各种工具和类
import torch.nn as nn
import numpy as np
# 用于执行神经网络中的各种操作，如激活函数、池化、归一化等
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pennylane as qml


# class Net(nn.Module):
#     # 输入状态和动作，当前例子中状态有2个表示为坐标(x,y)，动作有4个表示为(上下左右)
#     def __init__(self, n_states, n_actions):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(n_states, 10)  # 创建一个线性层，2行10列
#         self.fc2 = nn.Linear(10, n_actions)  # 创建一个线性层，10行4列
#         self.fc1.weight.data.normal_(0, 0.1) # 随机初始化生成权重，范围是0-0.1
#         self.fc2.weight.data.normal_(0, 0.1)
#
#     # 前向传播（用于状态预测动作的值）
#     def forward(self, state):
#         # 这里以一个动作为作为观测值进行输入(输入张量)
#         state = self.fc1(state)  # 线性变化后输出给10个神经元，格式：(x,x,x,x,x,x,x,x,x,x,x)
#         state = F.relu(state)  # 激活函数，将负值设置为零，保持正值不变
#         out = self.fc2(state)  # 经过10个神经元运算过后的数据，线性变化后把每个动作的价值作为输出。
#         return out



import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.dev = qml.device("default.qubit", wires=n_states)
        self.fc = nn.Linear(n_states, n_actions)
        self.fc.weight.data.normal_(0, 0.1)
        self.weights = nn.Parameter(torch.randn(n_states, n_states, 3))
        #定义了模型的权重参数 self.weights，它是一个 PyTorch 的可学习参数（nn.Parameter），初始化为一个大小为 (n_states, n_states, 3) 的张量，其中 n_states 表示量子电路的输入和输出的维度，3 可能表示某种特定的参数数量或者特征的维度

    def quantum_circuit(self, inputs, weights):
        @qml.qnode(self.dev)  #circuit函数将被转换为一个量子电路
        def circuit(inputs, weights):
            AngleEmbedding(inputs, wires=range(self.n_states)) #调用了一个量子门（Quantum Gate）AngleEmbedding，它将输入inputs编码到量子比特（Quantum Bit）上。wires=range(self.n_states)指定了将输入编码到的量子比特范围
            StronglyEntanglingLayers(weights, wires=range(self.n_states))  #调用了一个量子门StronglyEntanglingLayers，它对量子比特执行强纠缠层的操作  weights参数是强纠缠层的权重
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_states)] #量子电路的输出，它计算了每个量子比特的Pauli Z算符的期望值，并返回一个包含这些期望值的列表
        quantum_outputs = circuit(inputs, weights)  #调用了量子电路函数circuit，并传入输入inputs和权重weights，计算了量子电路的输出
        quantum_outputs_tensor = torch.tensor(quantum_outputs, dtype=torch.float) #将量子电路的输出转换为PyTorch张量，并指定了数据类型为torch.float
        return quantum_outputs_tensor #转换后的量子电路输出张量

    def forward(self, inputs):
        inputs = F.relu(inputs)
        inputs = torch.tensor(inputs, requires_grad=True) #将输入数据转换为PyTorch张量  在这些张量上需要计算梯度
        batch_size = inputs.shape[0]  # 获取输入张量的批量大小 用于后续的循环操作
        quantum_outputs = torch.zeros(batch_size, self.n_states) #创建了一个大小为(batch_size, self.n_states)的全零张量，用于存储量子电路的输出结果
        for i in range(batch_size):
            quantum_outputs[i] = self.quantum_circuit(inputs[i], self.weights) #对于每个输入数据，调用self.quantum_circuit方法，该方法接受输入数据和权重作为参数，计算量子电路的输出
        classical_outputs = self.fc(quantum_outputs) #将量子电路的输出作为输入，通过一个线性层（self.fc）进行处理，得到最终的经典输出结果
        return classical_outputs




# 定义DQN网络class
class DQN:
    #   n_states 状态空间个数；n_actions 动作空间大小
    def __init__(self, n_states, n_actions):
        print("<DQN init> n_states=", n_states, "n_actions=", n_actions)
        # 建立一个评估网络（即eval表示原来的网络） 和 Q现实网络 （即target表示用来计算Q值的网络）
        # DQN有两个net:target net和eval net,具有选动作、存储经验、学习三个基本功能
        self.eval_net, self.target_net = Net(n_states, n_actions), Net(n_states, n_actions)
        self.loss = nn.MSELoss()  # 损失均方误差损失函数
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.01) # 优化器，用于优化评估神经网络更新模型参数（仅优化eval），使损失函数尽量减小
        self.n_actions = n_actions  #   状态空间个数
        self.n_states = n_states    #   动作空间大小
        self.learn_step_counter = 0  # 用来记录学习到第几步了
        self.memory_counter = 0 # 用来记录当前指到数据库的第几个数据了
        # 创建一个2000行6列的矩阵，即表示可存储2000行经验，每一行6个特征值
        # 2*2表示当前状态state(x,y)和下一个状态next_state(x,y) + 1表示选择一个动作 + 1表示一个奖励值
        self.memory = np.zeros((2000, 2 * 2 + 1 + 1))
        self.cost = []  # 记录损失值
        self.steps_of_each_episode = []  # 记录每轮走的步数

    # 进行选择动作
    # state = [-0.5 -0.5]
    def choose_action(self, state, epsilon):
        # 扩展一行,因为网络是多维矩阵,输入是至少两维
        # torch.FloatTensor(x)先将x转化为浮点数张量
        # torch.unsqueeze(input, dim)再将一维的张量转化为二维的,dim=0时数据为行方向扩，dim=1时为列方向扩
        # 例如 [1.0, 2.0, 3.0] -> [[1.0, 2.0, 3.0]]
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        # 在大部分情况，我们选择 去max-value
        if np.random.uniform() < epsilon:   # greedy # 随机结果是否大于EPSILON（0.9）
            # 获取动作对应的价值
            action_value = self.eval_net.forward(state)
            #   torch.max() 返回输入张量所有元素的最大值，torch.max(input, dim)，dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
            #   torch.max(a, 1)[1] 代表a中每行最大值的索引
            #   data.numpy()[0] 将Variable转换成tensor
            # 哪个神经元值最大，则代表下一个动作
            #print(action_value)
            action = torch.max(action_value, 1)[1].data.numpy()[0]
            #print(action)
        # 在少部分情况，我们选择 随机选择 （变异）
        else:
            #   random.randint(参数1，参数2)函数用于生成参数1和参数2之间的任意整数，参数1 <= n < 参数2
            action = np.random.randint(0, self.n_actions)
        return action


    # 存储经验
    # 存储【本次状态，执行的动作，获得的奖励分，完成动作后产生的下一个状态】
    def store_transition(self, state, action, reward, next_state):
        # 把所有的记忆捆在一起，以 np 类型
        # 把 三个矩阵 s ,[a,r] ,s_  平铺在一行 [a,r] 是因为 他们都是 int 没有 [] 就无法平铺 ，并不代表把他们捆在一起了
        #  np.hstack()是把矩阵按水平方向堆叠数组构成一个新的数组
        transition = np.hstack((state, [action, reward], next_state))
        # index 是 这一次录入的数据在 MEMORY_CAPACITY 的哪一个位置
        # 如果记忆超过上线，我们重新索引。即覆盖老的记忆。
        index = self.memory_counter % 200
        self.memory[index, :] = transition  # 将transition添加为memory的一行
        self.memory_counter += 1


    # 从存储学习数据
    # target_net是达到次数后更新， eval_net是每次learn就进行更新
    def learn(self):
        # 更新 target_net，每循环30次更新一次
        if self.learn_step_counter % 30 == 0: # 将评估网络的参数状态复制到目标网络中 即将target_net网络变成eval_net网络，实现模型参数的软更新
            self.target_net.load_state_dict((self.eval_net.state_dict()))
        self.learn_step_counter += 1

        sample_index = np.random.choice(200, 16)  # 从[0,200)中随机抽取16个数据并组成一维数组，该数组表示记忆索引值
        memory = self.memory[sample_index, :]
        state = torch.FloatTensor(memory[:, :2])
        action = torch.LongTensor(memory[:, 2:3])
        reward = torch.LongTensor(memory[:, 3:4])
        next_state = torch.FloatTensor(memory[:, 4:6])
        q_eval = self.eval_net(state).gather(1, action)  #计算了在给定状态下采取的动作的预测 Q 值
        #DDQN部分
        max_actions=self.eval_net(next_state).max(1)[1].unsqueeze(1)  #使用评估网络计算了下一个状态 (next_state) 的 Q 值  找到了根据评估网络对下一个状态的 Q 值最大化的动作
        q_next=self.target_net(next_state).gather(1, max_actions) #使用目标网络 计算了下一个状态 (next_state) 的 Q 值
        q_target = reward + 0.9 * q_next.max(1)[0].unsqueeze(1)
        loss = self.loss(q_eval, q_target)
        # 记录损失值
        self.cost.append(loss.detach().numpy())
        # 根据误差，去优化我们eval网, 因为这是eval的优化器
        # 反向传递误差，进行参数更新
        self.optimizer.zero_grad()  # 梯度重置
        loss.backward()  # 反向求导
        self.optimizer.step()  # 更新模型参数

    # 绘制损失图
    def plot_cost(self):
        # np.arange(3)产生0-2数组
        #plt.plot(np.arange(len(self.cost)), self.cost)
        #plt.xlabel("step")
        #plt.ylabel("cost")
        #plt.show()
        file_path1 = "q_cost_ddqn.txt"
        #counts1=0
        for i in range(len(self.cost)):
            #counts1=counts1+1
            with open(file_path1,"a") as f:
                #f.write(f"{counts1} {self.cost[i]}\n")
                f.write(f"{self.cost[i]}\n")

    # 绘制每轮需要走几步
    def plot_steps_of_each_episode(self):
        #plt.plot(np.arange(len(self.steps_of_each_episode)), self.steps_of_each_episode)
        #plt.xlabel("episode")
        #plt.ylabel("done steps")
        #plt.show()
        file_path2 = "q_steps_ddqn.txt"
        #counts2 = 0
        for i in range(len(self.steps_of_each_episode)):
            #counts2 = counts2 + 1
            with open(file_path2, "a") as f:
                #f.write(f"{counts2} {self.steps_of_each_episode[i]}\n")
                f.write(f"{self.steps_of_each_episode[i]}\n")


