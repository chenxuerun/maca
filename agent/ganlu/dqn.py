import os

import torch.nn as nn
import torch
import torch.optim as optim

from agent.ganlu.cnn import ResUnit
from util.dl_util import mean_square
from util.ganlu_util import DQN_MODEL_FOLDER, RL_GAMMA, DIVIDE, NINE_ACTION

class DQN(nn.Module):
    def __init__(self, name, input_channels, preprocess_net=None):
        super(DQN, self).__init__()
        self.tau = 0.05
        self.preprocess = preprocess_net
        
        # 20 -> 10 -> 5 -> 3
        if NINE_ACTION:
            self.q1 = nn.Sequential(
                ResUnit(in_channels=input_channels, out_channels=2*input_channels, 
                    kernel_size=4, stride=2, padding=1, out_activation=nn.ReLU),
                ResUnit(in_channels=2*input_channels, out_channels=4*input_channels, 
                    kernel_size=4, stride=2, padding=1, out_activation=nn.ReLU),
                ResUnit(in_channels=4*input_channels, out_channels=1,
                    kernel_size=3, stride=2, padding=1, out_activation=None))
            self.q2 = nn.Sequential(
                ResUnit(in_channels=input_channels, out_channels=2*input_channels, 
                    kernel_size=4, stride=2, padding=1, out_activation=nn.ReLU),
                ResUnit(in_channels=2*input_channels, out_channels=4*input_channels, 
                    kernel_size=4, stride=2, padding=1, out_activation=nn.ReLU),
                ResUnit(in_channels=4*input_channels, out_channels=1,
                    kernel_size=3, stride=2, padding=1, out_activation=None))
            self.target_q1 = nn.Sequential(
                ResUnit(in_channels=input_channels, out_channels=2*input_channels, 
                    kernel_size=4, stride=2, padding=1, out_activation=nn.ReLU),
                ResUnit(in_channels=2*input_channels, out_channels=4*input_channels, 
                    kernel_size=4, stride=2, padding=1, out_activation=nn.ReLU),
                ResUnit(in_channels=4*input_channels, out_channels=1,
                    kernel_size=3, stride=2, padding=1, out_activation=None))
            self.target_q2 = nn.Sequential(
                ResUnit(in_channels=input_channels, out_channels=2*input_channels, 
                    kernel_size=4, stride=2, padding=1, out_activation=nn.ReLU),
                ResUnit(in_channels=2*input_channels, out_channels=4*input_channels, 
                    kernel_size=4, stride=2, padding=1, out_activation=nn.ReLU),
                ResUnit(in_channels=4*input_channels, out_channels=1,
                    kernel_size=3, stride=2, padding=1, out_activation=None))
        # 不变
        else:
            self.q1 = nn.Sequential(
                ResUnit(in_channels=input_channels, out_channels=input_channels,
                    kernel_size=5, stride=1, padding=2),
                ResUnit(in_channels=input_channels, out_channels=input_channels,
                    kernel_size=5, stride=1, padding=2),
                ResUnit(in_channels=input_channels, out_channels=input_channels,
                    kernel_size=3, stride=1, padding=1),
                ResUnit(in_channels=input_channels, out_channels=1,
                    kernel_size=3, stride=1, padding=1, out_activation=None))
            self.q2 = nn.Sequential(
                ResUnit(in_channels=input_channels, out_channels=input_channels,
                    kernel_size=5, stride=1, padding=2),
                ResUnit(in_channels=input_channels, out_channels=input_channels,
                    kernel_size=5, stride=1, padding=2),
                ResUnit(in_channels=input_channels, out_channels=input_channels,
                    kernel_size=3, stride=1, padding=1),
                ResUnit(in_channels=input_channels, out_channels=1,
                    kernel_size=3, stride=1, padding=1, out_activation=None))
            self.target_q1 = nn.Sequential(
                ResUnit(in_channels=input_channels, out_channels=input_channels,
                    kernel_size=5, stride=1, padding=2),
                ResUnit(in_channels=input_channels, out_channels=input_channels,
                    kernel_size=5, stride=1, padding=2),
                ResUnit(in_channels=input_channels, out_channels=input_channels,
                    kernel_size=3, stride=1, padding=1),
                ResUnit(in_channels=input_channels, out_channels=1,
                    kernel_size=3, stride=1, padding=1, out_activation=None))
            self.target_q2 = nn.Sequential(
                ResUnit(in_channels=input_channels, out_channels=input_channels,
                    kernel_size=5, stride=1, padding=2),
                ResUnit(in_channels=input_channels, out_channels=input_channels,
                    kernel_size=5, stride=1, padding=2),
                ResUnit(in_channels=input_channels, out_channels=input_channels,
                    kernel_size=3, stride=1, padding=1),
                ResUnit(in_channels=input_channels, out_channels=1,
                    kernel_size=3, stride=1, padding=1, out_activation=None))
        
        self.opt = optim.SGD(self.parameters(), lr=3e-4)
        self.model_path = os.path.join(DQN_MODEL_FOLDER, name)

    def save_model(self):
        if not os.path.exists(DQN_MODEL_FOLDER):
            os.mkdir(DQN_MODEL_FOLDER)
        state_dict = {'q1': self.q1.state_dict(), 'q2': self.q2.state_dict(),
            'target_q1': self.target_q1.state_dict(), 'target_q2': self.target_q2.state_dict(),
            'opt': self.opt.state_dict()}
        torch.save(state_dict, self.model_path)

    def load_model(self):
        device = torch.device('cuda')
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.q1.load_state_dict(checkpoint['q1'])
            self.target_q1.load_state_dict(checkpoint['target_q1'])
            self.q2.load_state_dict(checkpoint['q2'])
            self.target_q2.load_state_dict(checkpoint['target_q2'])
            self.opt.load_state_dict(checkpoint['opt'])
        self.to(device)

    def forward(self, state): # (n, 3, D, D)
        state = self.pre_process(state)
        q1 = self.q1(state)
        q2 = self.q2(state)
        return torch.where(q1 < q2, q1, q2)

    def pre_process(self, state):       # (n, 5, d, d)
        if self.preprocess:
            all_state = state[:, [0, 1, 2, 3]]
            all_state = self.preprocess(all_state)
            state = torch.cat([all_state, torch.unsqueeze(state[:, 4], dim=1)], dim=1)
        return state

    def sync_weight(self):
        for old, new in zip(self.target_q1.parameters(), self.q1.parameters()):
            old.data.copy_(old.data * (1 - self.tau) + new.data * self.tau)
        for old, new in zip(self.target_q2.parameters(), self.q2.parameters()):
            old.data.copy_(old.data * (1 - self.tau) + new.data * self.tau)

    # s和s_prime都传进来真实的，就是为了获得一个准确的Q
    def learn(self, s, a, r, s_prime, is_last):               # s: (n, 5, D, D) a: (n, 1, D, D) r: (n,)
        s = self.pre_process(s)
        q1 = self.q1(s)                                                            # (n, 1, D, D)
        q1 = q1[a==1]                                                             # (n,)
        q2 = self.q2(s)
        q2 = q2[a==1]
        with torch.no_grad():
            s_prime = self.pre_process(s_prime)          # (n, 9, D, D)
            if NINE_ACTION:
                q1_prime = self.target_q1(s_prime).reshape((-1, 9))     # (n, 1, D, D) -> (n, D*D)
                q2_prime = self.target_q2(s_prime).reshape((-1, 9))
            else:
                q1_prime = self.target_q1(s_prime).reshape((-1, DIVIDE * DIVIDE))
                q2_prime = self.target_q2(s_prime).reshape((-1, DIVIDE * DIVIDE))
            q_prime = torch.where(q1_prime < q2_prime, q1_prime, q2_prime)
            q_prime_max = q_prime.max(dim=1)[0]
            y = torch.where(is_last==1, r, r + RL_GAMMA * q_prime_max)

        q1_loss = mean_square(x=q1, l=y).mean()
        q2_loss = mean_square(x=q2, l=y).mean()

        self.opt.zero_grad()
        q1_loss.backward(retain_graph=True)
        q2_loss.backward()
        self.opt.step()
        self.sync_weight()