import gym
from tensorflow.keras.models import load_model
import numpy as np

def evaluate_dqn(env_name="CartPole-v1", model_path="dqn_model.h5", episodes=10):
    """
    Evaluate a trained DQN model.
    :param env_name: Name of the gym environment.
    :param model_path: Path to the saved model.
    :param episodes: Number of evaluation episodes.
    """
    env = gym.make(env_name)
    model = load_model(model_path)
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Uncomment the following line to visualize the evaluation
            # env.render()

            # Choose the action with the highest predicted Q-value
            action = np.argmax(model.predict(np.expand_dims(state, axis=0), verbose=0))
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

        rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward}")

    env.close()
    avg_reward = np.mean(rewards)
    print(f"Average Reward over {episodes} episodes: {avg_reward:.2f}")
    return rewards

if __name__ == "__main__":
    evaluate_dqn()
