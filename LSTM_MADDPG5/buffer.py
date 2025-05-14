import numpy as np


class MultiAgentReplayBuffer:
    def __init__(self, max_size, state_dims, action_dims, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.state_dims = state_dims
        self.action_dims = action_dims


        self.reward_memory = np.zeros((self.mem_size, n_agents)) #存储奖励值[B, C]
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)  #存储结束条件[B, C]

        self.init_actor_memory()

    def init_actor_memory(self):
        self.state_memory = []
        self.new_state_memory = []
        self.action_memory = []

        for i in range(self.n_agents):
            self.state_memory.append(
                np.zeros((self.mem_size, self.state_dims)))
            self.new_state_memory.append(
                np.zeros((self.mem_size, self.state_dims)))
            self.action_memory.append(
                np.zeros((self.mem_size, self.action_dims)))

    def store_transition(self, raw_obs, action, reward,
                         raw_obs_, done):
        # this introduces a bug: if we fill up the memory capacity and then
        # zero out our actor memory, the critic will still have memories to access
        # while the actor will have nothing but zeros to sample. Obviously
        # not what we intend.
        # In reality, there's no problem with just using the same index
        # for both the actor and critic states. I'm not sure why I thought
        # this was necessary in the first place. Sorry for the confusion!

        # if self.mem_cntr % self.mem_size == 0 and self.mem_cntr > 0:
        #    self.init_actor_memory()

        index = self.mem_cntr % self.mem_size

        for agent_idx in range(self.n_agents):
            self.state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.action_memory[agent_idx][index] = action[agent_idx]

        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        rewards = self.reward_memory[batch]

        terminal = self.terminal_memory[batch]

        states = []
        states_ = []
        actions = []
        for agent_idx in range(self.n_agents):
            states.append(self.state_memory[agent_idx][batch])
            states_.append(self.new_state_memory[agent_idx][batch])
            actions.append(self.action_memory[agent_idx][batch])

        return states, actions, rewards, states_, terminal

    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True
