import random
import gym
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
from tqdm import tqdm
import pennylane as qml
import collections

class ReplayBuffer:
    def __init__(self, capacity) -> None:
        self.buffer = collections.deque(maxlen=capacity)  # 队列 先进先出
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))  # 将1 step的信息保存到队列

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)  # 在buffer中随机采样batchsize大小的数据
        state, action, reward, next_state, done = zip(*transitions)  # 将数据解压
        return torch.tensor(np.array(state)).to(self.device), torch.tensor(action).to(self.device), torch.tensor(
            reward).to(self.device), torch.tensor(np.array(next_state)).to(self.device), torch.tensor(done).to(
            self.device)  # 返回的都是一个batchsize大小的数据

    def size(self):
        return len(self.buffer)

def encode(n_qubits,inputs):
    for wire in range(n_qubits):
        qml.RX(inputs[wire],wires=wire)


def layer(n_qubits, y_weight, z_weight):
    for wire, y_weight in enumerate(y_weight):
        qml.RY(y_weight, wires=wire)

    for wire, z_weight in enumerate(z_weight):
        qml.RZ(z_weight, wires=wire)

    for wire in range(n_qubits):
        qml.CZ(wires=[wire, (wire + 1) % n_qubits])

def measure(n_qubits):  #计算输出量子态在某个可观测量下的期望
    return [
        qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),  #@ 符号表示这两个观测量之间的张量积
        qml.expval(qml.PauliZ(2) @ qml.PauliZ(3))
    ]


def get_model(n_qubits, n_layers):  # 量子比特的数量和神经网络的层数
    dev = qml.device('default.qubit', wires=n_qubits)  # 默认模拟器 指定了量子比特为n_qubits
    shapes = {  # 字典 y和z权重参数的形状
        "y_weights": (n_layers, n_qubits),
        "z_weights": (n_layers, n_qubits)
    }

    @qml.qnode(dev, interface='torch')  # 将量子电路函数 circuit 编译成 PyTorch 可执行的函数
    def circuit(inputs, y_weights, z_weights):  # inputs 是输入数据，y_weights 和 z_weights 分别表示量子神经网络中的 Y 和 Z 权重参数
        for layer_id in range(n_layers):  # 用于构建量子神经网络的多层结构
            if layer_id == 0:  # 第一层时 调用encode对于数据编码
                encode(n_qubits, inputs)
            layer(n_qubits, y_weights[layer_id], z_weights[layer_id])  # 该函数应用于量子神经网络中的每一层，并接受对应的 Y 和 Z 权重参数
        return measure(n_qubits)  # 返回对量子比特进行 Pauli-Z 测量的结果，由 measure 函数定义

    model = qml.qnn.TorchLayer(circuit,
                               shapes)  # 创建了一个 PyTorch 可用的量子神经网络模型，并将量子电路函数 circuit 以及参数形状信息 shapes 传递给了 TorchLayer 类
    return model


class QuantumNet(torch.nn.Module):  # 量子的Q神经网络 继承自 torch.nn.Module 类，表示它是一个 PyTorch 模型
    def __init__(self, n_layers) -> None:  # n_layers表示量子神经网络中的层数
        super(QuantumNet, self).__init__()
        self.n_qubits = 4  # 模型中量子比特的数量
        self.n_actions = 2  # 输出动作的数量
        self.q_layer = get_model(self.n_qubits, n_layers)
        self.w_input = torch.nn.init.normal_(torch.nn.Parameter(torch.Tensor(self.n_qubits)),
                                             mean=0)  # 初始化一个大小为self.n_qubits的张量 值服从正态分布 且可以训练 初始化了输入权重参数 w_input，采用了正态分布初始化方法，均值为 0
        self.w_output = torch.nn.init.normal_(torch.nn.Parameter(torch.Tensor(self.n_actions)),
                                              mean=90)  # 代码初始化了输出权重参数 w_output，同样采用了正态分布初始化方法，均值为 90。

    def forward(self, inputs):  # 用于定义输入数据如何经过网络层进行计算得到输出
        inputs = inputs * self.w_input  # 对输入数据乘以输入权重参数 w_input
        inputs = torch.atan(inputs)  # 对输入数据应用反正切函数
        outputs = self.q_layer(inputs)  # 将处理后的输入数据输入到量子神经网络模型 q_layer 中进行计算，得到输出
        # print(outputs)
        outputs = (1 + outputs) / 2  # 对输出进行缩放和平移，确保输出在 [0, 1] 的范围内
        outputs = outputs * self.w_output  # 对输出数据乘以输出权重参数 w_output
        # print(outputs)
        return outputs  # 返回处理后的输出数据

class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)




class DQN:
    ''' DQN算法,包括Double DQN '''
    def __init__(self,n_layers,learning_rate,gamma,epsilon_init,epsilon_decay,epsilon_min,target_update,device,dqn_type = 'VanillaDQN'):
        self.action_dim = action_dim
        #self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        self.qnet = QuantumNet(n_layers).to(device)
        self.target_q_net=QuantumNet(n_layers).to(device)
        #self.target_q_net = Qnet(state_dim, hidden_dim,self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(),lr=learning_rate)
        self.gamma = gamma
        self.epsilon = 0
        self.target_update = target_update
        self.epsilon_init = epsilon_init  # 探索衰减因子 初始值
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min  # 最小值
        self.count = 0
        self.dqn_type = dqn_type
        self.device = device

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(2)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.qnet(state).argmax().item()
        return action
    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.qnet(state).max().item()

    def update_epsilon(self,step):
        self.epsilon = max(self.epsilon_min, self.epsilon_min +(self.epsilon_init - self.epsilon_min) *self.epsilon_decay**step)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)
        q_values = self.qnet(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        if self.dqn_type == 'DoubleDQN': # DQN与Double DQN的区别
            max_action = self.qnet(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else: # DQN的情况
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.qnet.state_dict())  # 更新目标网络
        self.count += 1

n_layers = 5
lr = 1e-2
num_episodes = 5000
hidden_dim = 128
gamma = 0.98
epsilon_init = 1
epsilon_min = 0.01
epsilon_decay = 0.99
target_update = 10
buffer_size = 5000
minimal_size = 500
batch_size = 16
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
env_name = 'CartPole-v0'
env = gym.make(env_name)
random.seed(21)
np.random.seed(21)
env.seed(21)
# replay_buffer = ReplayBuffer(buffer_size)
torch.manual_seed(21)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
step=0
return_list = []
now_step=[]
max_q_value_list=[]
file_path="q-DDQN.txt"
replay_buffer = ReplayBuffer(buffer_size)
agent = DQN(n_layers, lr, gamma, epsilon_init, epsilon_decay, epsilon_min, target_update, device,'DoubleDQN')
max_q_value = 0
for i in range(30):
    with tqdm(total=int(num_episodes / 100), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 100)):
            episode_return = 0
            state = env.reset()
            done = False
            agent.update_epsilon(step)
            while not done:
                action = agent.take_action(state)
                max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995  # 平滑处理
                max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值
                next_state, reward, done, _ = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)
            with open(file_path,"a") as f:
                f.write(f"{step} {episode_return}\n")
            return_list.append(episode_return)
            now_step.append(step)
            step += 1  #下一轮的训练
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                        '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

# episodes_list = list(range(len(return_list)))
# mv_return = rl_utils.moving_average(return_list, 5)
# plt.plot(episodes_list, mv_return)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('Double DQN on {}'.format(env_name))
# plt.show()
#
# frames_list = list(range(len(max_q_value_list)))
# plt.plot(frames_list, max_q_value_list)
# plt.axhline(0, c='orange', ls='--')
# plt.axhline(10, c='red', ls='--')
# plt.xlabel('Frames')
# plt.ylabel('Q value')
# plt.title('Double DQN on {}'.format(env_name))
# plt.show()