import gym
import numpy as np
from dqn_agent import DQNAgent

def train_dqn(env_name="CartPole-v1", episodes=500, model_save_path="dqn_model.h5"):
    """
    Train the DQN agent and save the trained model.
    :param env_name: Name of the gym environment.
    :param episodes: Number of training episodes.
    :param model_save_path: Path to save the trained model.
    """
    env = gym.make(env_name)
    state_shape = env.observation_space.shape
    action_space = env.action_space.n
    agent = DQNAgent(state_shape=state_shape, action_space=action_space)

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            agent.replay_buffer.add((state, action, reward, next_state, done))
            agent.train()
            state = next_state

        if episode % 10 == 0:
            agent.update_target_network()

        print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.4f}")

    # Save the trained model
    agent.q_network.save(model_save_path)
    print(f"Training complete. Model saved to {model_save_path}")

if __name__ == "__main__":
    train_dqn()
