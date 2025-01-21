from common import load_data, CryptoTradingEnv
from dqn_agent import DQNAgent

def train_dqn(agent, env, episodes=100, max_steps=500):
    results = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add((state, action, reward, next_state, done))
            agent.train()
            total_reward += reward
            state = next_state
            if done:
                break
        results.append(total_reward)
        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")
        if episode % 10 == 0:
            agent.update_target_network()
    with open("dqn_results.txt", "w") as f:
        for r in results:
            f.write(f"{r}\n")

if __name__ == "__main__":
    data = load_data("btcusdt.csv")
    env = CryptoTradingEnv(data)
    agent = DQNAgent(state_shape=(env.window_size, env.data.shape[1]), action_space=env.action_space.n)
    train_dqn(agent, env)
