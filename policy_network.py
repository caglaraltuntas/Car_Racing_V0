import torch.nn as nn

class Policy_Network(nn.Module):

    def __init__(self):
        super(Policy_Network, self).__init__()

        hidden_layer_0 = 256
        hidden_layer_1 = 64
        action_size = 5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fully_connected1 = nn.Linear(1600, hidden_layer_0)  # 1600 if the image 70x70
        self.fully_connected2 = nn.Linear(hidden_layer_0, hidden_layer_1)
        self.fully_connected3 = nn.Linear(hidden_layer_1, action_size)

        # self.activation = nn.Tanh()
        self.activation = nn.ReLU()
        self.activation_last_layer = nn.Softmax()

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
        output = self.activation(output)
        output = self.fully_connected3(output)
        output = self.activation_last_layer(output)

        return output
