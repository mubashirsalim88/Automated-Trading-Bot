import os
import matplotlib.pyplot as plt

def load_results(file_path):
    """
    Loads rewards or results from a text file.
    :param file_path: Path to the results file (e.g., dqn_results.txt).
    :return: List of rewards.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, "r") as file:
        rewards = [float(line.strip()) for line in file]
    return rewards

def plot_results(rewards, title="DQN Training Rewards", save_path=None):
    """
    Plots rewards over episodes.
    :param rewards: List of rewards per episode.
    :param title: Title of the plot.
    :param save_path: If provided, saves the plot to this path.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Episode Rewards", color="blue")
    plt.axhline(0, color="red", linestyle="--", label="Break-Even Line")
    plt.xlabel("Episodes")
    plt.ylabel("Reward (Profit/Loss)")
    plt.title(title)
    plt.legend()
    plt.grid()

    if save_path:
        # Ensure the results folder exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # File containing rewards
    results_file = "dqn_results.txt"

    # Define the results folder and file name
    results_folder = "results"
    plot_file = os.path.join(results_folder, "training_rewards_plot.png")

    # Load rewards and visualize
    try:
        rewards = load_results(results_file)
        plot_results(rewards, save_path=plot_file)
    except FileNotFoundError as e:
        print(e)
