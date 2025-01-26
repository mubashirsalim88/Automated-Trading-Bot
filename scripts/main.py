import gym
import numpy as np
from dqn_agent import DQNAgent

def train_agent(env_name="CartPole-v1", episodes=500):
    """
    Train a DQN agent on the specified environment.
    :param env_name: Name of the gym environment.
    :param episodes: Number of training episodes.
    """
    # Initialize the environment and agent
    env = gym.make(env_name)
    state_shape = env.observation_space.shape
    action_space = env.action_space.n
    agent = DQNAgent(state_shape=state_shape, action_space=action_space)

    rewards_history = []  # To store episode rewards

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Uncomment the following line to visualize the environment during training
            # env.render()

            # Choose an action and interact with the environment
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            # Store the experience in the replay buffer
            agent.replay_buffer.add((state, action, reward, next_state, done))

            # Train the agent
            agent.train()

            # Update the state
            state = next_state

        # Update target network periodically (e.g., every 10 episodes)
        if episode % 10 == 0:
            agent.update_target_network()

        # Record the total reward for the episode
        rewards_history.append(episode_reward)
        print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.4f}")

    env.close()
    return rewards_history

if __name__ == "__main__":
    # Train the DQN agent
    rewards = train_agent()

    # Optionally, save the rewards history for analysis
    np.save("rewards_history.npy", rewards)
    print("Training completed. Rewards saved to 'rewards_history.npy'.")
