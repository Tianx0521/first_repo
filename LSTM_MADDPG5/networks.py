import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from networkx.algorithms.flow import dinitz


class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dim, other_state_dim, action_dim, other_action_dim, hidden_dim, fc1_dims, fc2_dims, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name) #网络参数保存位置

        self.slstm = nn.LSTM(input_size=other_state_dim, hidden_size=hidden_dim, batch_first=True)  #状态LSTM
        self.alstm = nn.LSTM(input_size=other_action_dim, hidden_size=hidden_dim, batch_first=True)  #动作LSTM

        self.fc1 = nn.Linear(hidden_dim+state_dim+hidden_dim+action_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.33)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, other_states, action, other_actions):
        _, (hs, _) = self.slstm(other_states)
        _, (ha, _) = self.alstm(other_actions)
        hs = hs.squeeze(0)
        ha = ha.squeeze(0)
        #inputs = T.cat([state, hs, action, ha], dim=1)
        inputs = T.cat([state, hs], dim=1)
        inputs = T.cat([inputs, action], dim=1)
        inputs = T.cat([inputs, ha], dim=1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)    
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file, map_location='cpu'))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dim, other_state_dim, hidden_dim, fc1_dims, fc2_dims,
                 n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = chkpt_dir+"/"+name #网络参数的存放位置

        self.slstm = nn.LSTM(input_size=other_state_dim, hidden_size=hidden_dim, batch_first=True)  # 状态LSTM

        self.fc1 = nn.Linear(state_dim+hidden_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.8)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state, other_state):
        _, (h, _) = self.slstm(other_state)
        h = h.squeeze(0)
        inputs = T.cat([state, h], dim=1)
        x = F.leaky_relu(self.fc1(inputs))
        x = F.leaky_relu(self.fc2(x))
        pi = nn.Softsign()(self.pi(x)) # [-1,1]

        return pi

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file, map_location='cpu'))

