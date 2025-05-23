import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = InceptionResnetV1(pretrained='vggface2').eval()
        self.emb.requires_grad_(False)
        
        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x1, x2):
        h1 = self.emb(x1)
        h2 = self.emb(x2)

        x = torch.cat([h1, h2], dim=1)
        return self.fc(x)
    