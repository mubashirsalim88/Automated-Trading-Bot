import os
import gym
import numpy as np
from dqn_agent import DQNAgent
from tensorflow.keras.models import save_model

def train_and_save_ensemble_models(env_name="CartPole-v1", episodes=500, num_models=5, model_save_dir="ensemble_models"):
    """
    Train multiple DQN models and save them as ensemble models.
    :param env_name: Name of the gym environment.
    :param episodes: Number of training episodes per model.
    :param num_models: Number of models in the ensemble.
    :param model_save_dir: Directory where the models will be saved.
    """
    # Create a directory to save the models if it doesn't exist
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    env = gym.make(env_name)
    state_shape = env.observation_space.shape
    action_space = env.action_space.n
    
    for model_idx in range(num_models):
        print(f"Training model {model_idx + 1}/{num_models}...")
        
        # Initialize the agent and model
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

            print(f"Model {model_idx + 1}, Episode {episode + 1}/{episodes}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.4f}")

        # Save the model
        model_save_path = os.path.join(model_save_dir, f"dqn_model_{model_idx + 1}.h5")
        agent.q_network.save(model_save_path)
        print(f"Model {model_idx + 1} saved to {model_save_path}")

    print(f"All {num_models} ensemble models have been trained and saved.")

if __name__ == "__main__":
    train_and_save_ensemble_models()
