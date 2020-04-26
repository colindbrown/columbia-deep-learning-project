"""Training module: this is where MuZero neurons are trained."""

import numpy as np
#import tensorflow_core as tf
import torch.nn.functional as F
import torch

from config import MuZeroConfig
from networks.network import BaseNetwork
from networks.shared_storage import SharedStorage
from training.replay_buffer import ReplayBuffer


def train_network(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer, epochs: int):
    network = storage.current_network
    optimizer = storage.optimizer
    optimizer.zero_grad()

    for _ in range(epochs):
        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        update_weights(optimizer, network, batch)
        storage.save_network(network.training_steps, network)


def update_weights(optimizer: torch.optim, network: BaseNetwork, batch):
    def scale_gradient(tensor, scale: float):
        """Copied original tensor for non-gradient version """
        return (1. - scale) * tensor.detach() + scale * tensor

    def loss():
        loss = 0
        image_batch, targets_init_batch, targets_time_batch, actions_time_batch, mask_time_batch, dynamic_mask_time_batch = batch
        # Initial step, from the real observation: representation + prediction networks
        representation_batch, value_batch, policy_batch = network.initial_model(list(image_batch))

        # Only update the element with a policy target
        target_value_batch, _, target_policy_batch = zip(*targets_init_batch)
        mask_policy = list(map(lambda l: bool(l), target_policy_batch))
        target_policy_batch = list(filter(lambda l: bool(l), target_policy_batch))
        policy_batch = policy_batch[mask_policy]# tf.boolean_mask(policy_batch, mask_policy)

        # Compute the loss of the first pass
        #loss += tf.math.reduce_mean(loss_value(target_value_batch, value_batch, network.value_support_size))
        #loss += tf.math.reduce_mean(
            #tf.nn.softmax_cross_entropy_with_logits(logits=policy_batch, labels=target_policy_batch))

        # Recurrent steps, from action and previous hidden state.
        for actions_batch, targets_batch, mask, dynamic_mask in zip(actions_time_batch, targets_time_batch,
                                                                    mask_time_batch, dynamic_mask_time_batch):
            target_value_batch, target_reward_batch, target_policy_batch = zip(*targets_batch)

            # Only execute BPTT for elements with an action
            representation_batch = representation_batch[dynamic_mask]
            target_value_batch = torch.tensor(target_value_batch)[mask]
            target_reward_batch = torch.tensor(target_reward_batch)[mask]
            # Creating conditioned_representation: concatenate representations with actions batch
            actions_batch = torch.nn.functional.one_hot(torch.tensor(actions_batch), network.action_size)

            # Recurrent step from conditioned representation: recurrent + prediction networks
            conditioned_representation_batch = torch.cat([representation_batch, actions_batch.float()], axis=1)
            representation_batch, reward_batch, value_batch, policy_batch = network.recurrent_model(conditioned_representation_batch)
            # Only execute BPTT for elements with a policy target
            target_policy_batch = [policy for policy, b in zip(target_policy_batch, mask) if b]
            mask_policy = list(map(lambda l: bool(l), target_policy_batch))
            target_policy_batch = torch.tensor([policy for policy in target_policy_batch if policy])
            policy_batch = policy_batch[mask_policy]

            # Compute the partial loss
            l = (torch.mean(loss_value(target_value_batch, value_batch, network.value_support_size)) +
                F.mse_loss(target_reward_batch, torch.squeeze(reward_batch)) +
                torch.mean(torch.sum(- target_policy_batch * F.log_softmax(policy_batch, -1), -1)))

            # Scale the gradient of the loss by the average number of actions unrolled
            gradient_scale = 1. / len(actions_time_batch)
            loss += scale_gradient(l, gradient_scale)

            # Half the gradient of the representation
            representation_batch = scale_gradient(representation_batch, 0.5)
        return loss

    loss=loss()
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(network.dynamic_network.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(network.policy_network.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(network.reward_network.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(network.value_network.parameters(), 5)
    # if list(network.dynamic_network.parameters()):
    #     print(list(network.dynamic_network.parameters()))
    #     print(network.training_steps)
        #breakpoint()
    optimizer.step()
    network.training_steps += 1


def loss_value(target_value_batch, value_batch, value_support_size: int):
    batch_size = len(target_value_batch)
    targets = torch.zeros((batch_size, value_support_size))
    sqrt_value = target_value_batch# torch.sqrt(target_value_batch) I don't know why it's using sqrt
    floor_value = torch.floor(sqrt_value).long()
    rest = sqrt_value - floor_value
    targets[range(batch_size), floor_value] = 1 - rest
    targets[range(batch_size), floor_value + 1] = rest


    return torch.mean(torch.sum(- targets * F.log_softmax(value_batch, -1), -1))
