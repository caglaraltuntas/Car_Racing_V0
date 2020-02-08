from .supervised_learning_agent import Supervised_Learning_Agent
from .dqn_agent import DQN_Agent
import gym
import torch

if __name__ == '__main__':


    # env = gym.make("Enduro-v0")
    env = gym.make("CarRacing-v0")
    env._max_episode_steps = 50000000
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    Tensor = torch.Tensor
    LongTensor = torch.LongTensor

    agent = Supervised_Learning_Agent(state_size=(70, 70, 1), action_size=5)
    #agent = DQN_Agent(state_size=(70, 70, 1), action_size=5, test_mode=False, enable_dueling_dqn=True)
    agent.load("Car_Racing_weights14.pth")

    total_time = 0
    state = env.reset()

    while True:

        env.render()
        action, action_index, q_values = agent.act(state)
        if total_time < 20:
            action = [0,0.5,0]

        next_state, reward, done, _ = env.step(action)

        state = next_state

        print("Total Time: ", total_time)
        print("Policy Vector: ", q_values)
        print("Action: ", action)
        print("Reward", reward)

        if done:
            break

        total_time += 1
