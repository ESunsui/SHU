import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(11, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 64),
            nn.Dropout(0.02),
            nn.Linear(64, 64),
            nn.Dropout(0.02),
            nn.Linear(64, 64),
            nn.Dropout(0.02),
            nn.Linear(64, 64),
            nn.Dropout(0.02),
            nn.Linear(64, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.squeeze(-1)
        return x


if __name__ == '__main__':
    print(Model())
