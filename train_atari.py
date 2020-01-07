import random
import numpy as np
import gym
import os

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *
import torch
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DQN Atari')
    parser.add_argument('--load-checkpoint-file', type=str, default=None, 
                        help='Where checkpoint file should be loaded from (usually results/checkpoint.pth)')
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4',
                        help='The env type')
    parser.add_argument('--dqn-type', type=str, default='nature',
                        help='The model type of dqn')
    parser.add_argument('--logdir', type=str, default='exp',
                        help='the dir to store log and model')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--num-steps', type=int, default=100000,
                        help='steps to train')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='number of transitions to optimize at the same time')
    parser.add_argument('--print-freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--monitor', action='store_true',default=False)
    parser.add_argument('--play', action='store_true', default=False)






    args = parser.parse_args()
    # If you have a checkpoint file, spend less time exploring
    if(args.load_checkpoint_file):
        eps_start= 0.01
    else:
        eps_start= 1

    hyper_params = {
        'seed': 42,  # which seed to use
        # 'env': 'PongNoFrameskip-v4',  # name of the game
        'replay-buffer-size': int(5e3),  # replay buffer size
        # 'learning-rate': 1e-4,  # learning rate for Adam optimizer
        'discount-factor': 0.99,  # discount factor
        # 'dqn_type':'neurips',
        # total number of steps to run the environment for
        # 'num-steps': int(1e6),
        # 'batch-size': 32,  # number of transitions to optimize at the same time
        'learning-starts': 10000,  # number of steps before learning starts
        'learning-freq': 1,  # number of iterations between every optimization step
        'use-double-dqn': True,  # use double deep Q-learning
        'target-update-freq': 1000,  # number of iterations between every target network update
        'eps-start': eps_start,  # e-greedy start threshold
        'eps-end': 0.01,  # e-greedy end threshold
        'eps-fraction': 0.1,  # fraction of num-steps
        # 'print-freq': 10
    }

    # check logdir exists
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)


    np.random.seed(hyper_params['seed'])
    random.seed(hyper_params['seed'])

    assert 'NoFrameskip' in args.env, 'Require environment with no frameskip'
    env = gym.make(args.env)
    env.seed(hyper_params['seed'])

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)

    if(args.monitor):
        if not os.path.exists('./video/'):
            os.makedirs('./video/')
        env = gym.wrappers.Monitor(
            env, './video/', video_callable=lambda episode_id: episode_id % 50 == 0, force=True)

    replay_buffer = ReplayBuffer(hyper_params['replay-buffer-size'])

    agent = DQNAgent(
        env.observation_space,
        env.action_space,
        replay_buffer,
        use_double_dqn=hyper_params['use-double-dqn'],
        lr=args.learning_rate,
        batch_size=args.batch_size,
        gamma=hyper_params['discount-factor'],
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        dqn_type=args.dqn_type
    )

    if(args.load_checkpoint_file):
        print(f'Loading a policy - { args.load_checkpoint_file } ')
        agent.policy_network.load_state_dict(
            torch.load(args.load_checkpoint_file))

    eps_timesteps = hyper_params['eps-fraction'] * \
        float(args.num_steps)
    episode_rewards = [0.0]

    if not args.play:

        state = env.reset()
        for t in range(args.num_steps):
            fraction = min(1.0, float(t) / eps_timesteps)
            eps_threshold = hyper_params['eps-start'] + fraction * \
                (hyper_params['eps-end'] - hyper_params['eps-start'])
            sample = random.random()

            if(sample > eps_threshold):
                # Exploit
                action = agent.act(state)
            else:
                # Explore
                action = env.action_space.sample()

            next_state, reward, done, info = env.step(action)
            agent.memory.add(state, action, reward, next_state, float(done))
            state = next_state

            episode_rewards[-1] += reward
            if done:
                state = env.reset()
                episode_rewards.append(0.0)

            if t > hyper_params['learning-starts'] and t % hyper_params['learning-freq'] == 0:
                agent.optimise_td_loss()

            if t > hyper_params['learning-starts'] and t % hyper_params['target-update-freq'] == 0:
                agent.update_target_network()

            num_episodes = len(episode_rewards)

            if done and args.print_freq is not None and len(episode_rewards) % args.print_freq == 0:
                mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
                print('********************************************************')
                print('steps: {}'.format(t))
                print('episodes: {}'.format(num_episodes))
                print('mean 100 episode reward: {}'.format(mean_100ep_reward))
                print('% time spent exploring: {}'.format(int(100 * eps_threshold)))
                print('********************************************************')
                torch.save(agent.policy_network.state_dict(), os.path.join(args.logdir,'checkpoint.pth'))
                np.savetxt(os.path.join(args.logdir,'rewards_per_episode.csv'), episode_rewards,
                           delimiter=',', fmt='%1.3f')

    else:
        print("Running trained model")
        state = env.reset()
        while True:
            action = agent.act(state)
            # action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            if(done):
                state = env.reset()
            env.render()