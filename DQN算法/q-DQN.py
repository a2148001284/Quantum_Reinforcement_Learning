
# coding: utf-8

# In[1]:


import gym
import collections
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pennylane as qml
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
#pennylane+PyTorch量子卷积神经网络实现

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #没有gpu的系统 直接用cpu

import warnings
warnings.filterwarnings('ignore') #忽视过程中一些无关紧要的报警

# In[2]:


"""定义经验回放池的类 包括加入数据和采样数据"""
class ReplayBuffer:
    def __init__(self,capacity) -> None:
        self.buffer=collections.deque(maxlen=capacity)  #队列 先进先出
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def add(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))  #将1 step的信息保存到队列

    def sample(self,batch_size):
        transitions=random.sample(self.buffer,batch_size)  #在buffer中随机采样batchsize大小的数据
        state, action, reward, next_state, done = zip(*transitions)  #将数据解压
        return torch.tensor(np.array(state)).to(self.device),torch.tensor(action).to(self.device) , torch.tensor(reward).to(self.device), torch.tensor(np.array(next_state)).to(self.device), torch.tensor(done).to(self.device)  #返回的都是一个batchsize大小的数据
    
    def size(self):
        return len(self.buffer)


# In[3]:


#神经网络部分  输入层 隐藏层 输出层
#shot定了了应该对电路进行多少次采样评估
#编码层
def encode(n_qubits,inputs):
    for wire in range(n_qubits):
        qml.RX(inputs[wire],wires=wire) #wires指定要在哪个量子比特上执行操作 前面inputs[wire]控制的是旋转的角度

#电路层 /纠缠
def layer(n_qubits,y_weight,z_weight):
    for wire,y_weight in enumerate(y_weight):
        qml.RY(y_weight,wires=wire)
    
    for wire,z_weight in enumerate(z_weight):
        qml.RZ(z_weight,wires=wire)
    
    for wire in range(n_qubits):
        qml.CZ(wires=[wire,(wire+1)%n_qubits])
        #在当前量子比特和下一个量子比特之间增加一个CZ门 两比特门 控制比特为|1⟩时，在目标比特上施加一个相位翻转操作
        #控制比特和目标比特 确保循环 最后一个比特和第一个比特相连

#测量
def measure(n_qubits):  #计算输出量子态在某个可观测量下的期望
    return [
        qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),  #@ 符号表示这两个观测量之间的张量积
        qml.expval(qml.PauliZ(2) @ qml.PauliZ(3))
    ]

#量子神经网络的模型
def get_model(n_qubits,n_layers):  #量子比特的数量和神经网络的层数
    dev=qml.device('default.qubit',wires=n_qubits)  #默认模拟器 指定了量子比特为n_qubits
    shapes={ #字典 y和z权重参数的形状
        "y_weights": (n_layers,n_qubits),
        "z_weights": (n_layers,n_qubits)
    }
    @qml.qnode(dev,interface='torch')  #将量子电路函数 circuit 编译成 PyTorch 可执行的函数
    def circuit(inputs,y_weights,z_weights): #inputs 是输入数据，y_weights 和 z_weights 分别表示量子神经网络中的 Y 和 Z 权重参数
        for layer_id in range(n_layers): #用于构建量子神经网络的多层结构
            if layer_id==0: #第一层时 调用encode对于数据编码
                encode(n_qubits,inputs)
            layer(n_qubits,y_weights[layer_id],z_weights[layer_id])  #该函数应用于量子神经网络中的每一层，并接受对应的 Y 和 Z 权重参数
        return measure(n_qubits)  #返回对量子比特进行 Pauli-Z 测量的结果，由 measure 函数定义
    
    model=qml.qnn.TorchLayer(circuit,shapes) #创建了一个 PyTorch 可用的量子神经网络模型，并将量子电路函数 circuit 以及参数形状信息 shapes 传递给了 TorchLayer 类
    return model


class QuantumNet(torch.nn.Module):  #量子的Q神经网络 继承自 torch.nn.Module 类，表示它是一个 PyTorch 模型
    def __init__(self,n_layers) -> None: #n_layers表示量子神经网络中的层数
        super(QuantumNet,self).__init__()
        self.n_qubits=4  #模型中量子比特的数量
        self.n_actions=2 #输出动作的数量
        self.q_layer=get_model(self.n_qubits,n_layers)
        self.w_input=torch.nn.init.normal_(torch.nn.Parameter(torch.Tensor(self.n_qubits)),mean=0) #初始化一个大小为self.n_qubits的张量 值服从正态分布 且可以训练 初始化了输入权重参数 w_input，采用了正态分布初始化方法，均值为 0
        self.w_output=torch.nn.init.normal_(torch.nn.Parameter(torch.Tensor(self.n_actions)),mean=90) #代码初始化了输出权重参数 w_output，同样采用了正态分布初始化方法，均值为 90。
    
    def forward(self,inputs):  #用于定义输入数据如何经过网络层进行计算得到输出
        inputs=inputs*self.w_input # 对输入数据乘以输入权重参数 w_input
        inputs=torch.atan(inputs)  #对输入数据应用反正切函数
        outputs=self.q_layer(inputs)  #将处理后的输入数据输入到量子神经网络模型 q_layer 中进行计算，得到输出
        #print(outputs)
        outputs=(1+outputs)/2 #对输出进行缩放和平移，确保输出在 [0, 1] 的范围内
        outputs=outputs*self.w_output  #对输出数据乘以输出权重参数 w_output
        #print(outputs)
        return outputs  #返回处理后的输出数据


# In[4]:


"""DQN算法"""
class DQN:
    def __init__(self,n_layers,learning_rate,gamma,epsilon_init,epsilon_decay,epsilon_min,target_update,device) -> None:
        self.qnet=QuantumNet(n_layers).to(device) #Q网络 这里是量子网络
        self.target_qnet=QuantumNet(n_layers).to(device)  #目标q网络
        self.optimizer=torch.optim.Adam(self.qnet.parameters(),lr=learning_rate)  #初始化一个Adam优化器  self.qnet.parameters()获取qnet的所有参数 learning_rate为学习率
        self.gamma=gamma  #折扣因子
        self.epsilon=0  # epsilon-贪婪策略
        self.epsilon_init=epsilon_init  #探索衰减因子 初始值
        self.epsilon_decay=epsilon_decay
        self.epsilon_min=epsilon_min #最小值
        self.target_update=target_update  #目标网络更新因子 多少次更新一下目标q网络和当前q网络
        self.count=0  #q网络更新次数
        self.device=device

    def take_action(self,state): #最终可以获得目标的action
        if random.random()<self.epsilon:
            action=np.random.randint(2)  #随机选取动作（0，1） np.random.randint(x) 默认生成0-x-1的一个整数
        else:
            state=torch.tensor(state,dtype=torch.float).to(self.device)
            action=self.qnet(state).argmax().item()  #选取q值最大的动作
        return action
    
    def update_epsilon(self,step):
        self.epsilon = max(self.epsilon_min, self.epsilon_min +(self.epsilon_init - self.epsilon_min) *self.epsilon_decay**step)

    def update(self,transition_dict):
        states=torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device) #view(-1, 1)表示调整列数为1张量的形状
        #actions = transition_dict['actions'].clone().detach()
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)
        q_values=self.qnet(states).gather(1,actions)  #提取对应动作的q值
        max_next_q_values = self.target_qnet(next_states).max(1)[0].view(-1, 1)  #贪心策略 来选择下一步的策略 max(1)的1表示沿着列维度输出 [0]为最大值
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  #如果done=1 则游戏终止 不会有后续状态
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets)) #损失函数 计算预测值q_values与目标值q_targets之间的损失差值
        self.optimizer.zero_grad()  #将优化器中的梯度清零
        dqn_loss.backward() #1.计算损失关于模型的梯度 2.梯度会被保存 直到调用step方法才会更新模型参数
        self.optimizer.step()
        
        if self.count%self.target_update==0:
            self.target_qnet.load_state_dict(self.qnet.state_dict()) #将qnet的字典加载到target_qnet中
        self.count+=1


# In[2]:

n_layers = 5
lr = 1e-3
num_episodes = 5000
hidden_dim = 128
gamma = 0.98
epsilon_init = 1
epsilon_min = 0.01
epsilon_decay = 0.99
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_name = "CartPole-v0"
env = gym.make(env_name)
random.seed(21) # Python 标准库中的 random 模块的随机数生成器的种子为 21 数值为钥匙 相同数值的种子随机抽样的结果一致 确保每次抽样的结果都一样
np.random.seed(21)  #设置了 NumPy 库的随机数生成器的种子为 21
env.seed(21)  #设置了 Gym 环境的随机数生成器的种子为 21
torch.manual_seed(21)  #设置 PyTorch 库中的随机数生成器的种子为 21
replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQN(n_layers, lr, gamma, epsilon_init, epsilon_decay, epsilon_min, target_update, device)
return_list = []
#exp_name = datetime.now().strftime("DQN-%d_%m_%Y-%H_%M_%S")
#if not os.path.exists('./logs/'):
#    os.makedirs('./logs/')
#log_dir = './logs/{}/'.format(exp_name)
#os.makedirs(log_dir)
#writer = SummaryWriter(log_dir=log_dir)  #初始化了一个用于写入日志的对象 writer，并指定了日志存储的目录为 log_dir
step = 0  #第几轮的训练
total_reward = []
now_step = []
file_path="1.txt"

for i in range(30): #一共有30个任务大组
    with tqdm(total=int(num_episodes / 100), desc='Iteration %d' % i) as pbar: #每组任务有num_episodes / 100个 表示每次迭代完成百分之一的任务 Iteration x 表示是第几个任务组的迭代任务的标签
        for i_episode in range(int(num_episodes / 100)): #共50次整轮的训练 总共有30*50=1500次整轮的训练
            episode_return = 0
            state = env.reset()
            done = False
            agent.update_epsilon(step)
            while not done:  #开始不断的与环境交互且更新 直到done完成了
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)  # 采样一个batsize大小的的数据
                    transition_dict = {
                        # transition_dict是一个字典（键值对）
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)  # 用这些数据来更新q网络
            return_list.append(episode_return) #本轮结束以后的reward总值
            #writer.add_scalar('train/' + 'reward', episode_return, step)
            total_reward.append(episode_return)
            now_step.append(step)
            with open(file_path,"a") as f:
                f.write(f"{step} {episode_return}\n")
            step += 1  #下一轮的训练
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({ #设置了进度条的附加信息，用于显示当前迭代的周期数和最近十个周期的平均回报值。
                    'episode':
                        '%d' % (num_episodes / 100 * i + i_episode + 1),
                    'return':
                        '%.3f' % np.mean(return_list[-10:]) #最近十个周期的平均回报值 保留三位小数
                })
            pbar.update(1) #进度条的完成度增加1


#读入文件部分
def reads():
    list1=[]
    list2=[]
    with open(file_path,"r") as f:
        for line in f:
            items = line.strip().split()
            list1.append(int(items[0]))
            list2.append(items[1]) #如果是char类型而不是int



