import numpy as np
from maddpg import MADDPG
from wy_env import UAVEnv
from buffer import MultiAgentReplayBuffer
import time
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
from PIL import Image

warnings.filterwarnings('ignore')


def obs_list_to_state_vector(obs):  # 生成联合状态
    state = np.hstack([np.ravel(o) for o in obs])
    return state


def save_image(env_render, filename):  # 图片保存
    # Convert the RGBA buffer to an RGB image
    image = Image.fromarray(env_render, 'RGBA')  # Use 'RGBA' mode since the buffer includes transparency
    image = image.convert('RGB')  # Convert to 'RGB' if you don't need transparency

    image.save(filename)


if __name__ == '__main__':

    env = UAVEnv(num_uav=3)  # 实例化无人机环境
    # print(env.info)
    n_agents = env.num_uav  # 获取环境中的智能体数目
    state_dim = env.state_dim
    other_state_dim = env.other_state_dim
    action_dim = env.action_dim
    other_action_dim = env.other_action_dim
    hidden_dim = 64

    # action space is a list of arrays, assume each agent has same action space
    n_actions = 2  # 动作维度
    # maddpg_agents中包含了所有agent
    maddpg_agents = MADDPG(state_dim=state_dim-2, other_state_dim=other_state_dim, hidden_dim=hidden_dim, action_dim=action_dim, other_action_dim=other_action_dim,
                           n_agents=n_agents, n_actions=n_actions,
                           fc1=128, fc2=128,
                           alpha=0.0001, beta=0.0003, scenario='UAV_Round_up1',
                           chkpt_dir='./tmp/lstm_maddpg/')
    # 经验回放池  max_size, state_dim, other_state_dim, action_dim, other_action_dim, n_agents, batch_size
    #memory = MultiAgentReplayBuffer(max_size=100000, state_dim=state_dim, other_state_dim=other_state_dim, action_dim=action_dim, other_action_dim=other_action_dim, n_agents=n_agents, batch_size=64)
    memory = MultiAgentReplayBuffer(max_size=100000,state_dims=state_dim, action_dims=action_dim, n_agents=n_agents, batch_size=256)

    PRINT_INTERVAL = 100  #
    N_GAMES = 50001
    MAX_STEPS = 100
    total_steps = 0
    score_history = []  #
    target_score_history = []  #
    evaluate = False  # 是否可视化
    best_score = -600
    if evaluate:
        maddpg_agents.load_checkpoint()
        print('----  evaluating  ----')
    else:
        print('----training start----')

    for episode in range(N_GAMES):
        #print("第"+str(episode)+"回合    mem_size"+str(memory.mem_cntr))
        obs = env.reset()
        score = 0
        score_target = 0
        dones = [False] * n_agents
        episode_step = 0
        while not any(dones):
            if evaluate:
                # env.render()
                env_render = env.render()
                if episode_step % 10 == 0:
                    # Save the image every 10 episode steps
                    filename = f'images/episode_{episode}_step_{episode_step}.png'
                    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Create directory if it doesn't exist
                    save_image(env_render, filename)
                # time.sleep(0.01)

            actions = maddpg_agents.choose_action(obs, total_steps, evaluate)


            # #生成其他动作other_actions-----------------------------------------------------------------------------------------
            # other_actions = []
            # for i in range(n_agents):
            #     ai = []
            #     for j in order_index[i]:
            #         if i == j:
            #             continue
            #         ai.append(actions[j])
            #     other_actions.append(ai)
            #----------------------------------------------------------------------------------------------------------------

            obs_, rewards, dones = env.step(actions)



            if episode_step >= MAX_STEPS:  # 达到环境的最大执行步数
                dones = [True] * n_agents

            #state, other_state, action, other_action, reward, state_, other_state_, done
            memory.store_transition(raw_obs=obs, action=actions, reward=rewards, raw_obs_=obs_, done=dones)

            if total_steps % 10 == 0 and not evaluate:
                maddpg_agents.learn(memory, total_steps)

            obs = obs_
            score += sum(rewards[0:n_agents])
            score_target += rewards[-1]
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        target_score_history.append(score_target)
        avg_score = np.mean(score_history[-100:])
        avg_target_score = np.mean(target_score_history[-100:])
        if not evaluate:
            if episode % PRINT_INTERVAL == 0 and episode > 0 and avg_score > best_score:
                print('New best score', avg_score, '>', best_score, 'saving models...')
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if episode % PRINT_INTERVAL == 0 and episode > 0:
            print('episode', episode, 'average score {:.1f}'.format(avg_score),
                  '; average target score {:.1f}'.format(avg_target_score))

        # 保存网络参数

    maddpg_agents.save_checkpoint_last()

    # save data
    file_name = 'score_history_lstm.csv'
    if not os.path.exists(file_name):
        pd.DataFrame([score_history]).to_csv(file_name, header=False, index=False)
    else:
        with open(file_name, 'a') as f:
            pd.DataFrame([score_history]).to_csv(f, header=False, index=False)