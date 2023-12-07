import math
import sys

import numpy as np
from policy import Policy

class ValueFunctionWithApproximation(object):
    def __init__(self, num_features):
        """
        Initialize the value function approximation with a given number of features.

        input:
            num_features: The number of features in the feature vector
        """
        self.weights = np.zeros(num_features)

    def __call__(self,s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        return np.dot(self.weights, s)

    def update(self,alpha,G,s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        value_estimate = self(s_tau)
        delta = G - value_estimate
        self.weights += alpha * delta * s_tau

def semi_gradient_n_step_td(
    env, #open-ai environment
    gamma:float,
    pi:Policy,
    n:int,
    alpha:float,
    V:ValueFunctionWithApproximation,
    num_episode:int,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: #episodes to iterate
    output:
        None
    """
    for episode in range(num_episode):
        state = env.reset()
        next_state = None
        states = [state]
        rewards = [0]  # Reward for initial state
        T = sys.maxsize

        for t in range(T):
            if t < T:
                action = pi.action(state)
                next_state, reward, done, _ = env.step(action)
                states.append(next_state)
                rewards.append(reward)

                if done:
                    T = t + 1

            tau = t - n + 1  # Time step being updated
            if tau >= 0:
                G = sum([gamma**(i-tau-1) * rewards[i] for i in range(tau+1, min(tau+n, T)+1)])
                if tau + n < T:
                    G += gamma**n * V(states[tau+n])
                V.update(alpha, G, states[tau])

            if tau == T - 1:
                break

            state = next_state
