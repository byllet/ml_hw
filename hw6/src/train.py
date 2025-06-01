import torch
from torch import nn
import torch.optim as optim
import random
import numpy as np
import time
import json
import wandb
from pathlib import Path

from game.wrapped_flappy_bird import GameState
import params
from utils import image_to_tensor, transform_image

def train(model, start):
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.MSELoss()
    game_state = GameState()
    replay_memory = []

    action = torch.zeros([params.ACTION_NUM], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = transform_image(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    epsilon = params.MAX_EPS
    iteration = 0

    epsilon_decrements = np.linspace(params.MAX_EPS, params.MIN_EPS, params.ITER_NUM)
    logs = []
    wandb.init(project="hw6", name="train")
    while iteration < params.ITER_NUM:
        output = model(state)[0]

        action = torch.zeros([params.ACTION_NUM], dtype=torch.float32)
        if torch.cuda.is_available():
            action = action.cuda()

        random_action = random.random() <= epsilon
        
        action_index = [torch.randint(params.ACTION_NUM, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]
        if torch.cuda.is_available():
            action_index = action_index.cuda()

        action[action_index] = 1

        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = transform_image(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        replay_memory.append((state, action, reward, state_1, terminal))

        if len(replay_memory) > params.MEM_SIZE:
            replay_memory.pop(0)

        epsilon = epsilon_decrements[iteration]

        minibatch = random.sample(replay_memory, min(len(replay_memory), params.BATCH_SIZE))

        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()

        output_1_batch = model(state_1_batch)

        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + params.GAMMA * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        q_value = torch.sum(model(state_batch) * action_batch, dim=1)
        optimizer.zero_grad()
        y_batch = y_batch.detach()

        loss = criterion(q_value, y_batch)
        loss.backward()
        if iteration % 100 == 0:
            wandb.log({
                "loss": loss.item(),
                "reward": reward.item(),
                "Q_max": output.max().item(),
        })        
        optimizer.step()

        state = state_1
        iteration += 1
        wandb.log({"Q max": float(np.max(output.cpu().detach().numpy()))}, step=iteration)
        if iteration % 10000 == 0 or iteration == params.ITER_NUM - 1:
            torch.save(model.state_dict(), "pretrained_model/current_model_" + str(iteration + 2000000) + ".pt") # TODO: я тут кривыми ручками натрогал, поправить! 
            state_dict = {"iteration": iteration, "elapsed time": time.time() - start, "epsilon": epsilon, 
                          "reward": float(reward.numpy()[0][0]), "Q max": float(np.max(output.cpu().detach().numpy()))}
            logs.append(state_dict)
    log_file = Path("logs.json")
    with open(log_file, "a") as f:
        json.dump(logs, f)
                

        '''print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
              action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
              np.max(output.cpu().detach().numpy()))'''