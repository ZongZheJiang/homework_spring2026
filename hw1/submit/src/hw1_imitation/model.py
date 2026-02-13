"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
import numpy as np
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], chunk_size * action_dim))
        layers.append(nn.Unflatten(1, (chunk_size, action_dim))) 
        self.network = nn.Sequential(*layers)
        
    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # Forward pass through the network
        predicted_chunk = self.network(state)
        predicted_chunk = predicted_chunk.view(action_chunk.shape)

        # Compute MSE loss between predicted and actual action chunks
        criterion = nn.MSELoss()
        return criterion(predicted_chunk, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        
        for step in range(num_steps):
            # Generate action chunk for the current state
            action_chunk = self.network(state)
            # Here you would typically update the state based on the action taken
            # For simplicity, we will just return the action chunk
        return action_chunk
        raise NotImplementedError


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###

    # In Flow Matching, the model is the vector field predicting the velocity
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)        
        layers = []
        layers.append(nn.Linear(state_dim + action_dim * chunk_size + 1, hidden_dims[0]))  # +1 for time dimension
        layers.append(nn.ReLU())
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], chunk_size * action_dim))
        layers.append(nn.Unflatten(1, (chunk_size, action_dim))) 
        self.network = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        # Sample flow matching time of size [batch_size, 1] uniformly from [0, 1]
        flow_matching_time = torch.rand((batch_size, 1), device=state.device)

        # Flatten action chunk to shape [batch_size, chunk_size * action_dim]
        action_chunk = action_chunk.view(batch_size, -1)

        # Sample noise of size [batch_size, chunk_size * action_dim]
        noise = torch.randn_like(action_chunk)

        # Interpolate between action chunk and noise
        interpolated_actions = (1 - flow_matching_time) * noise + flow_matching_time * action_chunk

        # Compute the target velocity
        velocity = action_chunk - noise
        velocity = velocity.view(batch_size, -1)

        # print("interpolated_actions shape:", interpolated_actions.shape)
        # print("flow_matching_time shape:", flow_matching_time.shape)
        # print("state_chunk shape:", state_chunk.shape)

        # Predict the velocity using the network
        predicted_velocity = self.network(torch.cat([interpolated_actions, flow_matching_time, state], dim=1)).view(batch_size, -1)

        criterion = nn.MSELoss()
        return criterion(predicted_velocity, velocity)
        raise NotImplementedError

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # Forward Euler integration using the learned vector field
        batch_size = state.shape[0]
        chunk_size = self.chunk_size
        action_dim = self.action_dim
        current_action_estimate = torch.randn((batch_size, chunk_size, action_dim), device=state.device)        
        for step in range(num_steps):
            flat_actions = current_action_estimate.reshape(batch_size, -1)
            flow_matching_time = torch.full((batch_size, 1), step / num_steps, device=state.device)

            # print("noise shape:", noise.shape)
            # print("flow_matching_time shape:", flow_matching_time.shape)
            # print("state_chunk shape:", state_chunk.shape)
            current_action_estimate = current_action_estimate + (1.0 / num_steps) * self.network(torch.cat([flat_actions, flow_matching_time, state], dim=1))

        return current_action_estimate
        raise NotImplementedError


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
