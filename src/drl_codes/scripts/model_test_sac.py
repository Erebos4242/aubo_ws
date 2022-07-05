#!/usr/bin/env python3
from operator import imod
import core
from myrobotenv import *

env = RobotSim()
channel, width, height = 1, 100, 100
obs_dim = channel, width, height
act_dim = 4
model = core.MLPActorCritic(width, height, act_dim)
model.load_state_dict(torch.load('/home/ljm/data/saved_net/sac/1800model.txt'))


def get_action(o, deterministic=False):
    return model.act(torch.as_tensor(o, dtype=torch.float32),
                deterministic)


test_time = 30
for i in range(test_time):
    for i in range(10):
        o = env.get_states()
            # save_states(o, 'before')
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy.
        a = get_action(torch.unsqueeze(o, dim=0))
        env.step(a)
