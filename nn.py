import torch
import torch.nn as nn
import torch.optim as optim

from algo import ValueFunctionWithApproximation


class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self,
                 state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        self.model = nn.Sequential(
            # first hidden layer that transforms state_dims into 32 node space
            nn.Linear(state_dims, 32),
            nn.ReLU(),
            # 2nd hidden layer of 32 nodes
            nn.Linear(32, 32),
            nn.ReLU(),
            # value function should have no activation function
            nn.Linear(32, 1)
        )

        # Use AdamOptimizer with beta1=0.9, beta2=0.999 and learning rate alpha 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))

    def __call__(self,s):
        # Set your model to eval mode with model.eval()
        self.model.eval()
        # convert input from numpy to float tensor
        s = torch.from_numpy(s).float().unsqueeze(0)
        value = self.model(s)
        return value.item()

    def update(self,alpha,G,s_tau):
        # Set it back to train mode with model.train()
        self.model.train()
        # call optimizer.zero_grad()
        self.optimizer.zero_grad()

        # convert input from numpy to float tensor
        s_tau = torch.from_numpy(s_tau).float().unsqueeze(0)
        # Convert G to torch tensor
        G = torch.tensor(G, dtype=torch.float32)

        value_estimate = self.model(s_tau)
        # MSLE
        loss = (G - value_estimate).pow(2)
        loss.backward()
        # Update the model weights
        self.optimizer.step()

