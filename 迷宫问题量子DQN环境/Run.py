
from MazeEnv import Maze
from RL import DQN
import time

def run_maze(model):
    print("====Game Start====")
    step = 0    # 已进行多少步
    max_episode = 1500   # 总共需要进行多少轮
    success_count = 0
 
    for episode in range(max_episode):
        # flag=1
        state = env.reset()  # 环境和位置重置，但是memory一直保留
        #print("state:",state)
        step_every_episode = 0 # 本轮已进行多少步
        epsilon = episode / max_episode  # 动态变化随机值
 
        # 开始实验循环
        # 只有env认为 这个实验死了，才会结束循环
        while True:
            # if episode < 10:
            #     time.sleep(0.1)
            # if episode > 480:
            #     time.sleep(0.2)
            env.render()  # 刷新环境状态，显示新位置
            action = model.choose_action(state, epsilon)  # 根据输入的环境特征 s  输出选择动作 a
            next_state, reward, terminal = env.step(action) # env.step(a) 是执行 a 动作
            model.store_transition(state, action, reward, next_state)  # 模型存储经历
            # 按批更新
            if step > 50 and step % 5 == 0:
                model.learn()  #让预测网络先进行更新
            state = next_state  # 状态转变
            # if episode % 50 == 0 and episode!=0 and flag==1:
            #     model.plot_steps_of_each_episode()
            #     flag = 0
            # 状态是否为终止
            if terminal:
                if reward == 1:
                    print("本次学习轮次是episode=", episode, end=",")  # 第几轮
                    print("本轮此学习经过的步骤是step=", step_every_episode)  # 第几步
                    success_count=success_count+1
                    if(step_every_episode)>125:
                        step_every_episode=125
                    model.steps_of_each_episode.append(step_every_episode) # 记录每轮走的步数
                # model.steps_of_each_episode.append(step_every_episode)
                break
 
            step += 1   # 总步数+1
            step_every_episode += 1 # 当前轮的步数+1
 
    # 游戏环境结束
    print("====Game Over====")
    env.destroy()
 
 
if __name__ == "__main__":
    env = Maze()  # 环境
    # 实例化这个强化学习网络
    model = DQN(n_states=env.n_states,n_actions=env.n_actions)
    run_maze(model)  # 训练
    env.mainloop()  # mainloop()方法允许程序循环执行,并进入等待和处理事件
    model.plot_cost()  # 画误差曲线
    model.plot_steps_of_each_episode()  # 画每轮走的步数