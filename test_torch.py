


#展示如何使用训练好的模型并且进行可视化的展示
import gym
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import PG


#目前pytorch模型训练好以后的保存没有问题 但是会出现重新加载的时候会需要重新训练的情况 暂时没解决

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,device):
        self.policy_net = PolicyNet(state_dim, hidden_dim,action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.device = device

    def take_action(self, state):  # 根据动作概率分布随机采样
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.tensor([state_list[i]],
                                 dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G  # 每一步的损失函数
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降

    def GetNet(self):  #获得相应的训练网络
        return self.policy_net

env = gym.make('CartPole-v0')  # 创建OpenAI Gym环境
model = torch.load('Model/MyCartPole-v0.pth')
model.eval() #默认是训练模式 这样就可以变成评估模式 直接进行决策
# new_model=PolicyNet()
# new_model.load_state_dict(torch.load('./MyCartPole2-v0.pth'))
state = env.reset()  # 初始化环境
frames = []  # 存储渲染结果的列表
done = False  # 开始仿真

while not done:
    frame = env.render(mode='rgb_array') # 将当前环境渲染为RGB数组
    frames.append(Image.fromarray(frame))  # 将RGB数组转换为Image对象并添加到frames列表中
    state_tensor = torch.FloatTensor(state)  # 假设你已经有了一个训练好的模型model，并使用它进行动作选择
    action_probs = model(state_tensor.unsqueeze(0))
    action = torch.argmax(action_probs).item()
    state, _, done, _ = env.step(action) # 执行动作
    #如果你的模型输出的是动作的概率分布 通常可以使用torch.argmax()或者torch.argmax(action, dim=1)来找到概率最大的动作对应的索引
    #如果你的模型输出的是直接的动作值，那么你可以直接将这个值作为动作应用到环境中，不需要再使用np.argmax() 即env.step(action)
env.close()# 关闭环境
# 将frames列表中的图像序列保存为GIF动画
frames[0].save('cartpole_simulation.gif',save_all=True, append_images=frames[1:], optimize=False, duration=40, loop=0)


















