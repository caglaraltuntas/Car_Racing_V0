import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .dueling_neural_network import Dueling_Neural_Network


class DQN_Agent:

    def __init__(self, state_size, action_size, frame_skipping=False, test_mode=False,
                 enable_double_dqn=True, enable_dueling_dqn=False):

        self.state_size = state_size
        self.action_size = action_size
        self.frame_skipping = frame_skipping
        self.test_mode = test_mode
        self.memory_size = 50000
        self.memory = deque(maxlen=self.memory_size)
        # Once the deque is full. The oldest element in the deque is removed
        self.gamma = 0.99  # discount factor
        self.epsilon_decay = 0.99995
        self.epsilon_min = 0.1
        self.learning_rate = 0.00025

        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_dqn = enable_dueling_dqn
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.loss_func = nn.SmoothL1Loss()
        self.loss_func.cuda()

        if self.enable_dueling_dqn:
            self.model = Dueling_Neural_Network().to(self.device)
            self.target_model = Dueling_Neural_Network().to(self.device)

        else:
            self.model = Neural_Network().to(self.device)
            self.target_model = Neural_Network().to(self.device)

        if test_mode is True:
            self.epsilon = 0.01
        else:
            self.epsilon = 0.1  # epsilon-greedy (initial value)

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate)

    def preprocess_frame(self, rgb):

        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        frame = gray[10:80, 13:-13] / 255

        frame = torch.from_numpy(frame)

        frame = frame.to(self.device, dtype=torch.float32)
        frame = frame.unsqueeze(0).unsqueeze(0)

        return frame

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action_index, reward, next_state, done):  # Add done here

        # state = self.rgb2gray(state)
        # next_state = self.rgb2gray(next_state)

        self.memory.append((state, action_index, reward, next_state, done))  # and here

    def act(self, state):  # pass the distance value

        possible_actions = [[0, 0.5, 0],
                            [0, 0, 0.5],
                            [-0.25, 0, 0],
                            [0.25, 0, 0],
                            [0, 0, 0],
                            ]

        # if self.test_mode is False:

        if np.random.rand() <= self.epsilon:
            action_index = random.randrange(self.action_size)
            return possible_actions[action_index], action_index, \
                   0
        else:
            with torch.no_grad():
                state = self.preprocess_frame(state)
                act_values = self.model(state)
                action_index = torch.max(act_values, 1)[1].item()

            return possible_actions[action_index], action_index, \
                   act_values[0]  # The third argument is the q values

    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))

    def replay(self, batch_size):

        state, action, reward, next_state, done = self.sample(batch_size)

        state = [self.preprocess_frame(frame) for frame in state]
        state = torch.cat(state)

        next_state = [self.preprocess_frame(frame) for frame in next_state]

        next_state = torch.cat(next_state)

        reward = Tensor(reward).to(self.device)
        action = LongTensor(action).to(self.device)
        done = Tensor(done).to(self.device)

        if self.enable_double_dqn:
            next_state_indexes = self.model(next_state).detach()
            max_new_state_indexes = torch.max(next_state_indexes, 1)[1]

            next_state_values = self.target_model(next_state).detach()
            max_next_state_values = next_state_values.gather(1, max_new_state_indexes.unsqueeze(1)).squeeze(1)
        else:
            next_state_values = self.target_model(next_state).detach()
            max_next_state_values = torch.max(next_state_values, 1)[0]

        target_values = reward + (1 - done) * self.gamma * max_next_state_values
        predicted_values = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)

        loss = self.loss_func(predicted_values, target_values)
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        self.target_model.load_state_dict((torch.load(name)))
