import numpy as np

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable

from encoder import Encoder
from prepare_data import extract_sample, read_images

class ProtoNet(nn.Module):
    def __init__(self, encoder):
        """
        Args:trainx
            encoder : CNN encoding the images in sample
            n_way (int): number of classes in a classification task
            n_support (int): number of labeled examples per class in the support set
            n_query (int): number of labeled examples per class in the query set
        """
        super(ProtoNet, self).__init__()
        self.encoder = encoder

    def set_forward_loss(self, sample):
        """
        Computes loss, accuracy and output for classification task
        Args:
            sample (torch.Tensor): shape (n_way, n_support+n_query, (dim)) 
        Returns:
            torch.Tensor: shape(2), loss, accuracy and y_hat (predict)
        """
        sample_images = sample['images']
        n_way = sample['n_way']
        n_support = sample['n_support']
        n_query = sample['n_query']

        support_samples = sample_images[:, :n_support]
        query_samples = sample_images[:, n_support:]
        
        support_samples = support_samples.reshape(n_way * n_support, *support_samples.size()[2:])
        query_samples = query_samples.reshape(n_way * n_query, *query_samples.size()[2:])
        true_classes = np.array([[i] * n_query for i in range(n_way)]).ravel()

        support_samples = self.encoder.forward(support_samples)
        support_samples = support_samples.reshape(n_way, n_support, support_samples.size()[-1])
        query_samples = torch.squeeze(self.encoder.forward(query_samples))
        
        prototypes = torch.squeeze(support_samples.mean(1))
        
        sizes = (query_samples.size(0), prototypes.size(0), query_samples.size(1))
        prototypes = prototypes.unsqueeze(0).expand(*sizes)
        query_samples = query_samples.unsqueeze(1).expand(*sizes)
        distances = torch.pow(prototypes - query_samples, 2).sum(2)
        probabilities  = -distances
        true_classes = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
        true_classes = Variable(true_classes, requires_grad=False)
        log_softmax = F.log_softmax(probabilities, dim=1).view(n_way, n_query, -1)
        loss_val = -log_softmax.gather(2, true_classes).squeeze().view(-1).mean()
        _, y_hat = log_softmax.max(2)
        acc_val = torch.eq(y_hat, true_classes.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'y_hat': y_hat
            }
        
        
def load_protonet_conv(device, x_dim=(3, 28, 28), hid_dim=4, z_dim=64):
    """
    Loads the prototypical network model
    Arg:
      x_dim (tuple): dimension of input image
      hid_dim (int): dimension of hidden layers in conv blocks
      z_dim (int): dimension of embedded image
    Returns:
      Model (Class ProtoNet)
    """
    encoder = Encoder(device, x_dim, hid_dim, z_dim)

    return ProtoNet(encoder)

if __name__ == "__main__":
    trainx, trainy = read_images('images_background')
    sample_example = extract_sample(8, 5, 5, trainx, trainy)
    model = load_protonet_conv(device="cpu")
    print(model.set_forward_loss(sample_example))