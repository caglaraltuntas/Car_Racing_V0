import torch.nn as nn

class Dueling_Neural_Network(nn.Module):
    def __init__(self):
        super(Dueling_Neural_Network, self).__init__()

        hidden_layer_0 = 512
        hidden_layer_1 = 256
        hidden_layer_2 = 128
        action_size = 5

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.advantage1 = nn.Linear(1600, hidden_layer_0)
        self.advantage2 = nn.Linear(hidden_layer_0, action_size)
        # self.advantage3 = nn.Linear(hidden_layer_1, action_size)
        # self.advantage4 = nn.Linear(hidden_layer_2, action_size)

        self.value1 = nn.Linear(1600, hidden_layer_0)
        self.value2 = nn.Linear(hidden_layer_0, 1)
        # self.value3 = nn.Linear(hidden_layer_1, 1)
        # self.value4 = nn.Linear(hidden_layer_2, 1)
        self.activation1 = nn.Tanh()
        self.activation2 = nn.ReLU()

    def forward(self, x):
        output_conv = self.conv1(x)
        output_conv = self.activation2(output_conv)
        output_conv = self.conv2(output_conv)
        output_conv = self.activation2(output_conv)
        output_conv = self.conv3(output_conv)
        output_conv = self.activation2(output_conv)

        output_conv = output_conv.view(output_conv.size(0), -1)  # flatten

        output_advantage = self.advantage1(output_conv)
        output_advantage = self.activation2(output_advantage)
        output_advantage = self.advantage2(output_advantage)
        # output_advantage = self.activation2(output_advantage)
        # output_advantage = self.advantage3(output_advantage)
        # output_advantage = self.activation2(output_advantage)
        #  output_advantage = self.advantage4(output_advantage)

        output_value = self.value1(output_conv)
        output_value = self.activation2(output_value)
        output_value = self.value2(output_value)
        #  output_value = self.activation2(output_value)
        #   output_value = self.value3(output_value)
        # output_value = self.activation2(output_value)
        #  output_value = self.value4(output_value)

        output_final = output_value + output_advantage - output_advantage.mean()

        return output_final

class Neural_Network(nn.Module):
    def __init__(self):
        super(Neural_Network, self).__init__()

        hidden_layer_0 = 512
        hidden_layer_1 = 256
        action_size = 5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fully_connected1 = nn.Linear(1600, hidden_layer_0)
        self.fully_connected2 = nn.Linear(hidden_layer_0, action_size)


        # self.activation = nn.Tanh()
        self.activation = nn.ReLU()

    def forward(self, x):
        output_conv = self.conv1(x)
        output_conv = self.activation(output_conv)
        output_conv = self.conv2(output_conv)
        output_conv = self.activation(output_conv)
        output_conv = self.conv3(output_conv)
        output_conv = self.activation(output_conv)

        output_conv = output_conv.view(output_conv.size(0), -1)  # flatten

        output = self.fully_connected1(output_conv)
        output = self.activation(output)
        output = self.fully_connected2(output)

        return output