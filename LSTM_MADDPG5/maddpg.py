import os
import torch as T
import torch.nn.functional as F
from agent import Agent
import math_tool
import numpy as np
# from torch.utils.tensorboard import SummaryWriter

class MADDPG:
    def __init__(self, state_dim, other_state_dim, hidden_dim, action_dim, other_action_dim, n_agents, n_actions,
                 scenario='simple',  alpha=0.01, beta=0.02, fc1=128, 
                 fc2=128, gamma=0.99, tau=0.01, chkpt_dir='tmp/wymaddpg/'):
        self.agents = []  #智能体列表
        self.n_agents = n_agents  #智能体个数
        self.n_actions = n_actions  #智能体状态维度
        chkpt_dir += scenario   #生成存放网络参数的目录路径
        # self.writer = SummaryWriter(log_dir=os.path.join(chkpt_dir, 'logs'))

        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(state_dim=state_dim, other_state_dim=other_state_dim, hidden_dim=hidden_dim,
                                     action_dim=action_dim, other_action_dim=other_action_dim, n_actions=n_actions,
                                     agent_idx=1, chkpt_dir=chkpt_dir, alpha=alpha, beta=beta, fc1=fc1, fc2=fc2, gamma=gamma, tau=tau))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            os.makedirs(os.path.dirname(agent.actor.chkpt_file), exist_ok=True)
            agent.save_models()

    def save_checkpoint_last(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            os.makedirs(os.path.dirname(agent.actor.chkpt_file+"last"), exist_ok=True)
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, obs, time_step, evaluate):# timestep for exploration
        actions = []
        # TODO：根据观测值obs构建智能体的state和other_state,调用math_tool中的方法
        state, other_state = math_tool.compute_state_and_other_state_XD(obs=obs)
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(state=state[agent_idx], other_state=other_state[agent_idx], time_step=time_step, evaluate=evaluate)
            actions.append(action.tolist())
        return actions

    def learn(self, memory, total_steps):
        if not memory.ready():
            return

        obs, actions, rewards, obs_, dones = memory.sample_buffer()

        #TODO:计算states和other_states、other_states
        #1、将obs、actions、obs_的维度由[C, B, w]转换为[B, C, w]
        obs = np.transpose(np.array(obs), (1, 0, 2))
        actions = np.transpose(np.array(actions), (1, 0, 2))
        obs_ = np.transpose(np.array(obs_), (1, 0, 2))

        #2、调用math_tool中的方法生成states和other_states
        _, other_states = math_tool.compute_state_and_other_state_XD(obs, batchsize=memory.batch_size)
        states = obs
        other_states = np.transpose(np.array(other_states), (1, 0, 2, 3))

        _, other_states_ = math_tool.compute_state_and_other_state_XD(obs_, batchsize=memory.batch_size)
        states_ = obs_
        other_states_ = np.transpose(np.array(other_states_), (1, 0, 2, 3))

        other_actions = math_tool.compute_other_actions(actions=actions, batchsize=memory.batch_size)
        other_actions = np.transpose(np.array(other_actions), (1, 0, 2, 3))



        device = self.agents[0].actor.device

        states = T.tensor(states, dtype=T.float).to(device)  #[B, U, SD]
        other_states = T.tensor(other_states, dtype=T.float).to(device) #[B, U, U-1, OSD]
        actions = T.tensor(actions, dtype=T.float).to(device) #[B, U, w]
        other_actions = T.tensor(other_actions, dtype=T.float).to(device)  # [B, U, U-1, OAD]
        rewards = T.tensor(rewards, dtype=T.float).to(device) #[B, U]
        states_ = T.tensor(states_, dtype=T.float).to(device) #[B, U, SD]
        other_states_ = T.tensor(other_states_, dtype=T.float).to(device)  # [B, U, U-1, OSD]
        dones = T.tensor(dones).to(device) #[B, U]

        agents_new_actions = []
        agents_old_actions = []
    
        for agent_idx, agent in enumerate(self.agents):

            new_states = states[:, agent_idx, 2:] #T.tensor(actor_new_states[agent_idx],  dtype=T.float).to(device)
            agents_other_states = other_states[:, agent_idx, :, :]

            new_pi = agent.target_actor.forward(state=new_states, other_state=agents_other_states)

            agents_new_actions.append(new_pi)
            agents_old_actions.append(actions[:, agent_idx])



        for agent_idx, agent in enumerate(self.agents):
            with T.no_grad():
                #critic的输入：state, other_states, action, other_actions
                critic_value_ = agent.target_critic.forward(state=states_[:, agent_idx, 2:], other_states=other_states_[:, agent_idx,:, :], action=agents_new_actions[agent_idx], other_actions=other_actions[:, agent_idx,:, :]).flatten()
                target = rewards[:,agent_idx] + (1-dones[:,0].int())*agent.gamma*critic_value_

            critic_value = agent.critic.forward(state=states[:, agent_idx, 2:], other_states=other_states[:, agent_idx,:, :], action=agents_old_actions[agent_idx],  other_actions=other_actions[:, agent_idx,:, :]).flatten()
            
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()
            agent.critic.scheduler.step()

            mu_states = states[:, agent_idx, 2:] #T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
            #oa = agents_old_actions[agent_idx].clone()
            oa = agent.actor.forward(state=mu_states, other_state=other_states[:, agent_idx,:, :])
            actor_loss = -T.mean(agent.critic.forward(state=states[:, agent_idx, 2:], other_states=other_states[:, agent_idx,:, :], action=oa, other_actions=other_actions[:, agent_idx,:, :]).flatten())
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()
            agent.actor.scheduler.step()

            # self.writer.add_scalar(f'Agent_{agent_idx}/Actor_Loss', actor_loss.item(), total_steps)
            # self.writer.add_scalar(f'Agent_{agent_idx}/Critic_Loss', critic_loss.item(), total_steps)

            # for name, param in agent.actor.named_parameters():
            #     if param.grad is not None:
            #         self.writer.add_histogram(f'Agent_{agent_idx}/Actor_Gradients/{name}', param.grad, total_steps)
            # for name, param in agent.critic.named_parameters():
            #     if param.grad is not None:
            #         self.writer.add_histogram(f'Agent_{agent_idx}/Critic_Gradients/{name}', param.grad, total_steps)
            
        for agent in self.agents:    
            agent.update_network_parameters()
