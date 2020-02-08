import msvcrt
import gym
import torch
from .supervised_learning_agent import Supervised_Learning_Agent
if __name__ == '__main__':

    env = gym.make("CarRacing-v0")
    # env._max_episode_steps = 50000000
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    Tensor = torch.Tensor
    LongTensor = torch.LongTensor
    
    agent = Supervised_Learning_Agent(state_size=(70, 70, 1), action_size=5, test_mode=False)
    # agent.load("Car_Racing_weights1.pth")

    action_size = 5
    batch_size = 32
    n_episodes = 300
    episode_length = 5000
    update_frequency_target = 50
    update_frequency_main = 1
    hidden_layer = 512
    save_frequency = 1000
    user_drive = True

    output_dir = "Car_Racing_Weights"

    total_time = 0

    for episode in range(n_episodes):

        state = env.reset()
        # flag  = 0
        for time in range(episode_length):  # max time, increase this number later

            env.render()
            # action, action_index, q_values = agent.act(state)

            if total_time % 5 == 0:
                print("Choose an action:")

                if user_drive is True:
                    action_user = msvcrt.getch()
                    if action_user == b"w":
                        action_user = [0, 0.5, 0]
                        action_user_index = 0
                    elif action_user == b"s":
                        action_user = [0, 0, 0.5]
                        action_user_index = 1
                    elif action_user == b"a":
                        action_user = [-0.25, 0, 0]
                        action_user_index = 2
                    elif action_user == b"d":
                        action_user = [0.25, 0, 0]
                        action_user_index = 3
                    else:
                        action_user = [0, 0, 0]
                        action_user_index = 4

            next_state, reward, done, _ = env.step(action_user)

            agent.remember(state, action_user_index, total_time)
            state = next_state

            if total_time > 1000 and total_time % update_frequency_main == 0:
                agent.replay(batch_size)

            print("Total Time: ", total_time)
            print("Episode: ", episode)
            # print("Policy Vector: ", q_values)
            print("Action: ", action_user)
            print("Reward", reward)

            if done:
                break

            if (total_time + 1) % save_frequency == 0 and total_time > 1000:
                agent.save("Car_Racing_weights{}.pth".format(int(total_time / save_frequency)))

            total_time += 1
