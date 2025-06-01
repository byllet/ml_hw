import os
import sys
import time
import wandb
import torch
from game.wrapped_flappy_bird import GameState 
from utils import image_to_tensor, transform_image
from model import NeuralNetwork, init_weights
from train import train
import params


def test(model):
    game_state = GameState()

    action = torch.zeros([params.ACTION_NUM], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = transform_image(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)
    wandb.init(project="hw6", name="test")
    while True:
        output = model(state)[0]

        action = torch.zeros([params.ACTION_NUM], dtype=torch.float32)
        if torch.cuda.is_available():
            action = action.cuda()

        action_index = torch.argmax(output)
        if torch.cuda.is_available():
            action_index = action_index.cuda()
        action[action_index] = 1

        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = transform_image(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        state = state_1


def main(mode):
    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':
        model = NeuralNetwork()
        model.load_state_dict(torch.load('pretrained_model/current_model_2200000.pt'))
        '''model = torch.load(
            'pretrained_model/current_model_9999.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()'''

        if cuda_is_available:
            model = model.cuda()

        test(model)

    elif mode == 'train':
        wandb.init(project="hw6", name="baseline")
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

        model = NeuralNetwork()
        
        loading = True
        if loading:
            model.load_state_dict(torch.load('pretrained_model/current_model_2199999.pt'))

        if cuda_is_available:
            model = model.cuda()
        if not loading:
            model.apply(init_weights)
        start = time.time()

        train(model, start)


if __name__ == "__main__":
    main(sys.argv[1])