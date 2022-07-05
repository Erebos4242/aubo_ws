#!/usr/bin/env python3
from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import core as core
from myrobotenv import *
# from testenv import *
import torchvision.transforms as torch_tran
import time
import threading
import cv2
import json
import os


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, *obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, *obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}



def sac(actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=100000, gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        save_freq=1):

    torch.manual_seed(seed)
    np.random.seed(seed)

    # env = RobotSim()
    env = RobotSim()
    channel, width, height = 1, 100, 100
    obs_dim = channel, width, height
    act_dim = 4

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = 1.0

    # Create actor-critic module and target networks
    ac = actor_critic(width, height, act_dim, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)


    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()


        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True


        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32),
                      deterministic)

    # def get_states():
    #     _, image = env.get_img()
    #     image = image[:, 140:500]
    #     image = image[60: 420]
    #     # # cv2.imwrite('/home/ljm/data/temp.png', image)
    #
    #     image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #
    #     image = np.array([image])
    #     return torch.tensor(image)

    # def test_agent():
    #     for j in range(num_test_episodes):
    #         o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
    #         while not(d or (ep_len == max_ep_len)):
    #             # Take deterministic actions at test time
    #             o, r, d, _ = test_env.step(get_action(o, True))
    #             ep_ret += r
    #             ep_len += 1

    # Prepare for interaction with environment

    # def thread_update():
    #     for j in range(update_every):
    #         batch = replay_buffer.sample_batch(batch_size)
    #         update(data=batch)

    def save_states(obs, file_name):
        nonlocal img_count
        obs = torch.squeeze(obs)
        img = np.array(obs)
        img = img.astype(np.uint8)
        cv2.imwrite(f'/home/ljm/data/push_img/{img_count}_{file_name}.png', img)
        img_count += 1


    experience_dir_name = str(int(time.time()))
    os.mkdir(f'/home/ljm/data/experience/{experience_dir_name}')

    def save_experience():
        ptr = replay_buffer.ptr
        dir = f'/home/ljm/data/experience/{experience_dir_name}'
        np.save(f'{dir}/obs.npy', replay_buffer.obs_buf[:ptr])
        np.save(f'{dir}/act.npy', replay_buffer.act_buf[:ptr])
        np.save(f'{dir}/rew.npy', replay_buffer.rew_buf[:ptr])
        np.save(f'{dir}/obs2.npy', replay_buffer.obs2_buf[:ptr])
        np.save(f'{dir}/done.npy', replay_buffer.done_buf[:ptr])

    def load_experience():
        for dir_name in os.listdir('/home/ljm/data/experience/'):
            dir = f'/home/ljm/data/experience/{dir_name}'
            if not os.listdir(dir):
                continue
            obs = np.load(f'{dir}/obs.npy')
            act = np.load(f'{dir}/act.npy')
            rew = np.load(f'{dir}/rew.npy')
            obs2 = np.load(f'{dir}/obs2.npy')
            done = np.load(f'{dir}/done.npy')
            for i in range(len(obs)):
                replay_buffer.store(obs[i], act[i], rew[i], obs2[i], done[i])

    def judge_start_point(x, y):
        _, image_org = env.get_img()
        lenth_per_pixel = 0.00186
        x -= 0.5

        x_pixel = 240 - int(x / lenth_per_pixel)
        y_pixel = 320 + int(y / lenth_per_pixel)
        off_set = 6

        image = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, 210, 255, cv2.THRESH_BINARY_INV)

        # cv2.circle(image, (y_pixel, x_pixel), 10, (255, 255, 255), 1)
        # cv2.imshow('img', image)
        # cv2.waitKey(0)

        if image[x_pixel - off_set][y_pixel] == 255 or image[x_pixel + off_set][y_pixel] == 255 or \
                image[x_pixel][y_pixel - off_set] == 255 or image[x_pixel][y_pixel + off_set] == 255 or \
                image[x_pixel - 5][y_pixel - 5] == 255 or image[x_pixel + 5][y_pixel + 5] == 255 or \
                image[x_pixel - 5][y_pixel + 5] == 255 == 255 or image[x_pixel + 5][y_pixel - 5] == 255 == 255:
            return True
        else:
            return False

    start_time = time.time()

    num_episodes = 2000
    start_episodes = 100
    update_after = 100
    update_every = 30
    # Main loop: collect experience in env and update/log each epoch
    img_count = 0
    load_experience()

    for i_episode in range(1, num_episodes):
        print(f'episode: {i_episode}')
        env.reset()
        for i_step in range(10):
            o = env.get_states()
            # save_states(o, 'before')
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy.
            if i_episode > start_episodes:
                a = get_action(torch.unsqueeze(o, dim=0))
            else:
                x1 = np.random.uniform(0.2, 0.8)
                y1 = np.random.uniform(-0.3, 0.3)
                while judge_start_point(x1, y1):
                    x1 = np.random.uniform(0.2, 0.8)
                    y1 = np.random.uniform(-0.3, 0.3)

                x2 = np.random.uniform(0.2, 0.8)
                y2 = np.random.uniform(-0.3, 0.3)
                a = [x1, y1, x2, y2]

            # Step the env
            o2, d, r = env.step(a)
            # save_states(o2, 'after')
            print(f'step: {i_step}, reward: {r}')

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)

            # Store experience to replay buffer
            replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d:
                break
        print(f'done step: {i_step}')

        # Update handling
        if i_episode >= update_after and i_episode % update_every == 0:
            # th_update = threading.Thread(target=thread_update())
            # th_update.daemon = True
            # th_update.start()
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # # End of epoch handling
        # if (t+1) % steps_per_epoch == 0:
        #     epoch = (t+1) // steps_per_epoch
        #
        #     # Test the performance of the deterministic version of the agent.
        #     test_agent()
        if i_episode % 200 == 0:
            save_experience()
        if i_episode % 200 == 0:
            torch.save(ac_targ.state_dict(), f'/home/ljm/data/saved_net/sac/{i_episode}model.txt')
    end_time = time.time()
    print(end_time - start_time)
    torch.save(ac_targ.state_dict(), '/home/ljm/data/saved_net/sac/1000model.txt')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    torch.set_num_threads(torch.get_num_threads())

    sac(actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs)
