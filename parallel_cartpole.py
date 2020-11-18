import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
from agent import Agent, Policy
from cp_cont import CartPoleEnv
from parallel_env import ParallelEnvs
import pandas as pd


# Policy training function
def train(env_name, print_things=True, train_run_id=0, train_timesteps=200000, update_steps=50):
    # Create a Gym environment
    # This creates 64 parallel envs running in 8 processes (8 envs each)
    env = ParallelEnvs(env_name, processes=8, envs_per_process=8)

    # Get dimensionalities of actions and observations
    action_space_dim = env.action_space.shape[-1]
    observation_space_dim = env.observation_space.shape[-1]

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy)

    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []

    # Run actual training
    # Reset the environment and observe the initial state
    observation = env.reset()

    # Loop forever
    for timestep in range(train_timesteps):
        # Get action from the agent
        action, action_probabilities = agent.get_action(observation)
        previous_observation = observation

        # Perform the action on the environment, get new state and reward
        observation, reward, done, info = env.step(action.detach().numpy())

        for i in range(len(info["infos"])):
            env_done = False
            # Check if the environment is finished; if so, store cumulative reward
            for envid, envreward in info["finished"]:
                if envid == i:
                    reward_history.append(envreward)
                    average_reward_history.append(np.mean(reward_history[-500:]))
                    env_done = True
                    break
            # Store action's outcome (so that the agent can improve its policy)
            agent.store_outcome(previous_observation[i], observation[i],
                                action_probabilities[i], reward[i], env_done)

        if timestep % update_steps == update_steps-1:
            print(f"Update @ step {timestep}")
            agent.update_policy(0)

        plot_freq = 1000
        if timestep % plot_freq == plot_freq-1:
            # Training is finished - plot rewards
            plt.plot(reward_history)
            plt.plot(average_reward_history)
            plt.legend(["Reward", "500-episode average"])
            plt.title("AC reward history (non-episodic, parallel)")
            plt.savefig("rewards_%s.png" % env_name)
            plt.clf()
            torch.save(agent.policy.state_dict(), "model.mdl")
            print("%d: Plot and model saved." % timestep)
    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id]*len(reward_history),
                         # TODO: Change algorithm name for plots, if you want
                         "algorithm": ["Nonepisodic parallel AC"]*len(reward_history),
                         "reward": reward_history})
    torch.save(agent.policy.state_dict(), "model_%s_%d.mdl" % (env_name, train_run_id))
    return data


# Function to test a trained policy
def test(env_name, episodes, params, render):
    # Create a Gym environment
    env = gym.make(env_name)

    # Get dimensionalities of actions and observations
    action_space_dim = env.action_space.shape[-1]
    observation_space_dim = env.observation_space.shape[-1]

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    policy.load_state_dict(params)
    agent = Agent(policy)

    test_reward, test_len = 0, 0
    for ep in range(episodes):
        done = False
        observation = env.reset()
        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(observation, evaluation=True)
            observation, reward, done, info = env.step(action.detach().cpu().numpy())

            if render:
                env.render()
            test_reward += reward
            test_len += 1
    print("Average test reward:", test_reward/episodes, "episode length:", test_len/episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--env", type=str, default="ContinuousCartPole-v0", help="Environment to use")
    parser.add_argument("--train_timesteps", type=int, default=200000, help="Number of timesteps to train for")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    args = parser.parse_args()

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.
    if args.test is None:
        try:
            train(args.env, train_timesteps=args.train_timesteps)
        # Handle Ctrl+C - save model and go to tests
        except KeyboardInterrupt:
            print("Interrupted!")
    else:
        state_dict = torch.load(args.test)
        print("Testing...")
        test(args.env, 100, state_dict, args.render_test)

