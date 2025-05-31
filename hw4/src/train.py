import wandb
import torch
import torch.optim as optim
from tqdm.notebook import tnrange
from tqdm import tqdm

from model import load_protonet_conv, ProtoNet
from prepare_data import extract_sample, read_images

tqdm.get_lock().locks = []



def train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size):
    """
    Trains the protonet
    Args:
      model
      optimizer
      train_x (np.array): images of training set
      train_y(np.array): labels of training set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      max_epoch (int): max epochs to train on
      epoch_size (int): episodes per epoch
    """
    #divide the learning rate by 2 at each epoch, as suggested in paper
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
    epoch = 0  # epochs done so far
    stop = False  # status to know when to stop

    while epoch < max_epoch and not stop:
        running_loss = 0.0
        running_acc = 0.0

        with tqdm(total=epoch_size) as progress_bar:
          for episode in range(epoch_size):
              sample = extract_sample(n_way, n_support, n_query, train_x, train_y)
              optimizer.zero_grad()
              loss, output = model.set_forward_loss(sample)
              print(loss)
              running_loss += output['loss']
              running_acc += output['acc']
              loss.backward()
              optimizer.step()
              progress_bar.update()
              progress_bar.set_description('Loss = {:.5f}, Acc = {:.2f}'.format(output['loss'], output['acc']))

        
        epoch_loss = running_loss / epoch_size
        epoch_acc = running_acc / epoch_size

        progress_bar.refresh()
        epoch += 1
        scheduler.step()
        metrics = {'Loss': epoch_loss, 'Accuracy': epoch_acc}
        wandb.log(metrics, step=epoch)

if __name__ == "__main__":
    wandb.init(project="hw4", name="baseline")
    device = "cpu" 
    trainx, trainy = read_images('images_background')
    testx, testy = read_images('images_evaluation')
    model = load_protonet_conv("cpu")
    n_way = 60
    n_support = 5
    n_query = 5
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    max_epoch = 5
    epoch_size = 200
    train_x = trainx
    train_y = trainy

    train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size)
