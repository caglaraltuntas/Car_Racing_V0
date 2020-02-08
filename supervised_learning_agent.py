import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from .policy_network import Policy_Network


class Supervised_Learning_Agent:

    def __init__(self, state_size, action_size, frame_skipping=False, test_mode=False):

        self.state_size = state_size
        self.action_size = action_size
        self.frame_skipping = frame_skipping
        self.test_mode = test_mode
        self.memory_size = 50000
        self.memory = deque(maxlen=self.memory_size)
        # Once the deque is full. The oldest element in the deque is removed

        self.learning_rate = 0.001#0.00025

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.loss_func = nn.CrossEntropyLoss()
        self.loss_func.cuda()

        self.model = Policy_Network().to(self.device)

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate)

    def preprocess_frame(self, rgb):

        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        frame = gray[10:80, 13:-13] / 255
        frame = frame / 255

        frame = torch.from_numpy(frame)

        frame = frame.to(self.device, dtype=torch.float32)
        frame = frame.unsqueeze(0).unsqueeze(0)

        return frame

    def remember(self, state, action_index, total_time):  # Add done here

        self.memory.append((state, action_index, total_time))  # and here

    def act(self, state):  # pass the distance value

        possible_actions = [[0, 0.3, 0],
                            [0, 0, 0.5],
                            [-0.25, 0, 0],
                            [0.25, 0, 0],
                            [0, 0, 0],
                            ]

        # if self.test_mode is False:


        with torch.no_grad():

            state = self.preprocess_frame(state)
            act_values = self.model(state)
            #action_index = torch.max(act_values, 1)[1].item()
            act_values = act_values.data.cpu().numpy()[0]
            action_index = random.choices(range(self.action_size), k=1, weights=act_values)[0]

        return possible_actions[action_index], action_index, \
               act_values # The third argument is the policy

    def sample(self, batch_size):
        # dont enumerate the memory instead get the index values

        return zip(*random.sample(self.memory, batch_size))

    def replay(self, batch_size):

        state, action, index_value = self.sample(batch_size)

    #    state = [self.preprocess_frame(frame) for frame in state]
        state = torch.cat(state)

        action = torch.LongTensor(action).to(self.device)
        #done = Tensor(done).to(self.device)

        predicted_values = self.model(state)# .gather(1, action.unsqueeze(1)).squeeze(1)

        loss = self.loss_func(predicted_values, action)
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()


    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

