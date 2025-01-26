import os
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from common import create_environment

def evaluate_agent(agent, environment, num_episodes=10):
    """
    Evaluates the DQN agent on a given environment.
    :param agent: Trained DQN agent.
    :param environment: Trading environment.
    :param num_episodes: Number of episodes to evaluate.
    :return: List of cumulative rewards per episode.
    """
    cumulative_rewards = []

    for episode in range(num_episodes):
        state = environment.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state, explore=False)  # No exploration during evaluation
            next_state, reward, done, _ = environment.step(action)
            total_reward += reward
            state = next_state

        cumulative_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes}: Total Reward = {total_reward:.2f}")

    return cumulative_rewards

def plot_evaluation_results(cumulative_rewards, save_path=None):
    """
    Plots the evaluation results.
    :param cumulative_rewards: List of cumulative rewards per episode.
    :param save_path: If provided, saves the plot to this path.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(cumulative_rewards)), cumulative_rewards, color="green", alpha=0.7)
    plt.axhline(np.mean(cumulative_rewards), color="red", linestyle="--", label="Mean Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.title("DQN Agent Evaluation Results")
    plt.legend()
    plt.grid()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Evaluation plot saved to: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Paths
    model_path = "results/dqn_model.pth"  # Path to the trained model
    results_folder = "results"
    plot_file = os.path.join(results_folder, "evaluation_results_plot.png")

    # Create the trading environment
    environment = create_environment()

    # Load the trained agent
    agent = DQNAgent(state_size=environment.observation_space.shape[0],
                     action_size=environment.action_space.n)
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print("Trained model loaded successfully.")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Evaluate the agent
    num_episodes = 10
    cumulative_rewards = evaluate_agent(agent, environment, num_episodes=num_episodes)

    # Plot the evaluation results
    plot_evaluation_results(cumulative_rewards, save_path=plot_file)
