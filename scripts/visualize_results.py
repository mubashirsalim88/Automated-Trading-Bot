import matplotlib.pyplot as plt
import numpy as np

def visualize_training_rewards(rewards_file="rewards_history.npy"):
    """
    Visualize the rewards history from training.
    :param rewards_file: Path to the file containing rewards history.
    """
    # Load rewards history
    rewards = np.load(rewards_file)

    # Plot rewards
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Training Rewards Over Episodes")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    visualize_training_rewards()
