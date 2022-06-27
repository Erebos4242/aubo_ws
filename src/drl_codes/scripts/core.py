import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class CNN(nn.Module):

    def __init__(self,h, w):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.fc1 = nn.Linear(linear_input_size, 256)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return x


class SquashedGaussianMLPActor(nn.Module):
    def __init__(self,width, height, act_dim, hidden_sizes, activation):
        super().__init__()
        self.net = CNN(height, width)
        self.fc = nn.Linear(256, 256)
        self.mu_layer = nn.Linear(256, act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        net_out = F.relu(self.fc(net_out))
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self,width, height, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = CNN(height, width)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(260, 260)
        self.out = nn.Linear(260, 1)

    def forward(self, obs, act):
        x = self.q(obs)
        x = torch.squeeze(self.flat(x))
        x = torch.cat([x, act], dim=-1)
        x = F.relu(self.fc1(x))
        q_values = self.out(x)

        return torch.squeeze(q_values, -1)  # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, width, height, act_dim, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        act_dim = act_dim

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(width, height, act_dim, hidden_sizes, activation)
        self.q1 = MLPQFunction(width, height, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(width, height, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        act_limit = 0.3
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            a = a.numpy()[0]
            a[0] = a[0] * act_limit + 0.5
            a[1] = a[1] * act_limit
            a[2] = a[2] * act_limit + 0.5
            a[3] = a[3] * act_limit
            return a


# net = MLPActorCritic(100, 100, 4)
# print(net)
# o = torch.zeros(100, 1, 100, 100)
# a = torch.zeros(100, 4)
# x = net.q1(o, a)
# print(x.shape)
# print(x)