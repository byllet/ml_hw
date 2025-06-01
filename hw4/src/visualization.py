import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from prepare_data import read_images
from model import load_protonet_conv
from prepare_data import extract_sample
import torchvision

def visualize(sample, model):
    sample_images = sample['images'].to("cuda")
    n_way = sample['n_way']
    n_support = sample['n_support']
    n_query = sample['n_query']
    
    all_images = sample_images.reshape(-1, *sample_images.shape[2:])
    all_embeddings = model.encoder(all_images)
    
    support_emb = all_embeddings[:n_way*n_support]
    query_emb = all_embeddings[n_way*n_support:]
    
    prototypes = support_emb.reshape(n_way, n_support, -1).mean(dim=1)
    
    feats = torch.cat([
        prototypes.unsqueeze(1),
        query_emb.reshape(n_way, n_query, -1)
    ], dim=1).reshape(-1, prototypes.shape[-1]).cpu().detach().numpy()
    
    tsne = TSNE(n_components=2, perplexity=min(30, len(feats)//3))
    feats_2d = tsne.fit_transform(feats)
    
    plt.figure(figsize=(10, 8))
    for i in range(n_way):
        start = i * (n_query + 1)
        end = start + n_query + 1
        plt.scatter(feats_2d[start:end, 0], feats_2d[start:end, 1], label=f'Class {i}')    
    plt.legend()
    plt.title('Визуализация предсказания с тестирования')
    plt.show()

def display_sample(sample):
    """
    Displays sample in a grid
    Args:
      sample (torch.Tensor): sample of images to display
    """
    #need 4D tensor to create grid, currently 5D
    sample_4D = sample.view(sample.shape[0] * sample.shape[1], *sample.shape[2:])
    #make a grid
    out = torchvision.utils.make_grid(sample_4D, nrow=sample.shape[1])
    
    plt.figure(figsize=(16, 7))
    plt.imshow(out.permute(1, 2, 0))

if __name__ == "__main__":
    device = "cuda"  
    trainx, trainy = read_images('images_background')
    testx, testy = read_images('images_evaluation')
    model = load_protonet_conv("cuda")
    model.load_state_dict(torch.load("trained_model.pt"))
    model.to("cuda")
    n_way = 7
    n_support = 5
    n_query = 5
    test_sample = extract_sample(n_way, n_support, n_query, testx, testy)
    display_sample(test_sample['images'])
    visualize(test_sample, model)
