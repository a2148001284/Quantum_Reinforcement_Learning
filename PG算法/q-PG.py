
# coding: utf-8

# In[3]:


import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pennylane as qml
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

import warnings
warnings.filterwarnings('ignore') #忽视过程中一些无关紧要的报警


# In[4]:


#编码层
def encode(n_qubits,inputs):
    for wire in range(n_qubits):
        qml.RX(inputs[wire],wires=wire)

#电路层
def layer(n_qubits,y_weight,z_weight):
    for wire,y_weight in enumerate(y_weight):
        qml.RY(y_weight,wires=wire)
    
    for wire,z_weight in enumerate(z_weight):
        qml.RZ(z_weight,wires=wire)

def entangle(n_qubits):
    for wire in range(n_qubits):
        qml.CZ(wires=[wire,(wire+1)%n_qubits])

#测量
def measure(n_qubits):
    return [
        qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
        qml.expval(qml.PauliZ(2) @ qml.PauliZ(3))
    ]

#量子电路
def get_model(n_qubits,n_layers):
    dev=qml.device('default.qubit',wires=n_qubits)
    shapes={
        "y_weights": (n_layers+1,n_qubits),
        "z_weights": (n_layers+1,n_qubits)
    }
    @qml.qnode(dev,interface='torch')
    def circuit(inputs,y_weights,z_weights):
        for layer_id in range(n_layers):
            encode(n_qubits,inputs)
            layer(n_qubits,y_weights[layer_id],z_weights[layer_id])
            entangle(n_qubits)
        layer(n_qubits,y_weights[n_layers],z_weights[n_layers])
        return measure(n_qubits)
    model=qml.qnn.TorchLayer(circuit,shapes)
    return model

class PolicyNet(torch.nn.Module):  #策略网络 其输入是某个状态，输出则是该状态下的动作概率分布
    def __init__(self,n_layers) -> None:
        super(PolicyNet,self).__init__()
        self.n_qubits=4
        self.n_actions=2
        self.q_layer=get_model(self.n_qubits,n_layers)
        self.w_input=torch.nn.init.normal_(torch.nn.Parameter(torch.Tensor(self.n_qubits)))
        self.w_output=torch.nn.init.normal_(torch.nn.Parameter(torch.Tensor(self.n_actions)))
        #self.o_input=torch.nn.Parameter(torch.tensor([[-1.0 if i % 2 == 0 else 1.0 for i in range(self.n_actions)]]), requires_grad=True)

    
    def forward(self,inputs):
        inputs=inputs*self.w_input
        inputs=torch.atan(inputs)
        outputs=self.q_layer(inputs)
        outputs=self.w_output*outputs
        outputs=torch.nn.functional.softmax(outputs,dim=1)  #将输出转换为概率分布，以便于分类任务的处理
        return outputs


# In[5]:


class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 device):
        self.policy_net = PolicyNet(5).to(device)
        self.gamma = gamma  # 折扣因子
        self.device = device
        self.build()

    def build(self): #构建优化器及其参数设置
        params=[]
        params.append({'params': self.policy_net.parameters(),'lr': 0.001})
        params.append({'params': self.policy_net.w_input, 'lr': 0.01})
        params.append({'params': self.policy_net.w_output, 'lr': 0.01})
        #params.append({'params': self.policy_net.o_input, 'lr': 0.01})
        self.optimizer=torch.optim.Adam(self.policy_net.parameters())

    def take_action(self, state):  # 根据当前的状态来选择动作
        state = torch.tensor([state], dtype=torch.float).to(self.device) #将状态转化为张量
        probs = self.policy_net(state) #probs 是一个包含了动作的概率分布的张量，例如 [0.2, 0.5, 0.3]，表示有三个动作，对应的概率分别为 0.2、0.5 和 0.3
        action_dist = torch.distributions.Categorical(probs) #对离散型随机变量进行建模，其中每个动作对应一个离散的概率。在强化学习中，通常使用这样的分布来表示在给定状态下，选择每个动作的概率分布
        action = action_dist.sample()  #使用 sample() 方法来从该分布中采样一个动作
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards'] #从给定的 transition_dict 字典中获取了回报列表，其中保存了每个时间步的回报值
        state_list = transition_dict['states']  #保存了每个时间步的状态
        action_list = transition_dict['actions']  #保存了每个时间步的选择动作
        running_add = 0  #用于累积未来的回报
        for i in reversed(range(len(reward_list))): #逆向遍历回报列表的循环，从最后一个时间步开始往前
            if reward_list[i] == 0:  #当前时间步的回报为 0，意味着这是一个终止状态，将 running_add 置为 0
                running_add = 0
            else:
                running_add = running_add * self.gamma + reward_list[i] #计算了当前时间步的回报的累积值，乘以折扣因子 gamma 后加上当前时间步的回报
                reward_list[i] = running_add #将计算得到的累积回报值保存回到原来的 reward_list 中，以便后续使用

        # 标准化奖励
        reward_mean = np.mean(reward_list)  #计算了回报列表中所有回报的均值，用于后续的标准化计算
        reward_std = np.std(reward_list)  #计算了回报列表中所有回报的标准差，同样用于标准化
        for i in range(len(reward_list)):
            #对每个时间步的回报进行标准化处理，即将每个回报减去均值，然后除以标准差，得到标准化后的回报值
            reward_list[i] = (reward_list[i] - reward_mean) / reward_std

        self.optimizer.zero_grad() #优化器中之前积累的梯度清零
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.tensor([state_list[i]],dtype=torch.float).view(1,-1).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action)) #计算当前状态下选择当前动作的对数概率，即策略网络对应动作的log概率值
            loss = -log_prob * reward  #计算当前时间步的损失值，即对数概率与标准化后的回报值的乘积的负值
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降 使用优化器更新策略网络的参数，根据计算得到的梯度进行参数更新


# In[6]:


learning_rate = 1e-3
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
env_name = "CartPole-v0"
env = gym.make(env_name)
np.random.seed(21)
env.seed(21)
torch.manual_seed(21)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma,
                  device)

return_list = []
#exp_name = datetime.now().strftime("PG-%d_%m_%Y-%H_%M_%S")
#if not os.path.exists('./logs/'):
#    os.makedirs('./logs/')
#log_dir = './logs/{}/'.format(exp_name)
#os.makedirs(log_dir)
#writer = SummaryWriter(log_dir=log_dir)
total_reward = []
now_step = []
file_path="2.txt"
episode_count=0
def opt():
    plt.figure()
    plt.plot(total_reward)
    plt.xlabel('Epsiode')
    plt.ylabel('Collected rewards')
    plt.show()
for i in range(100):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            if episode_return<=200:
                total_reward.append(episode_return)
                now_step.append(episode_count)
                with open(file_path,"a") as f:
                    f.write(f"{episode_count} {episode_return}\n")
            #writer.add_scalar('train/' + 'reward',episode_return, episode_count)
            episode_count+=1
            agent.update(transition_dict)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)
        #opt()

