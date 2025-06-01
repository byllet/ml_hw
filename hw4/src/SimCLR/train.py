from pathlib import Path
from time import gmtime, strftime
import yaml

import numpy as np
from tqdm import tqdm

import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader

from simclr import  BaseTrainProcess, ProjectionHead
from data import load_datasets, get_datasets
from sklearn.metrics import accuracy_score


class ClassifierCLR(nn.Module):
    def __init__(self, encoder, emb_size):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.projector = ProjectionHead(emb_size, 2048, 964) #964 - число букв из всех алфавитов background

    def forward(self, x):
        out = self.encoder(x)
        xp = self.projector(torch.squeeze(out))
        return xp

    def make_requires_grad(self, requires_grad):
        for p in self.encoder.parameters():
            p.requires_grad = requires_grad


class ClassifierCLRTrainer:
    def __init__(self, model, hyp, train_loader, valid_loader):
        self.best_loss = 1e100
        self.best_acc = 0.0
        self.current_epoch = -1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = model.to(self.device)

        self.hyp = hyp
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self._init_model()

    
    def _init_model(self):
        model_params = [params for params in self.model.parameters() if params.requires_grad]
        self.optimizer = torch.optim.AdamW(model_params, lr=self.hyp['lr'], weight_decay=self.hyp['weight_decay'])

        # "decay the learning rate with the cosine decay schedule without restarts"
        self.warmupscheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: (epoch + 1) / 10.0)
        self.mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            500,
            eta_min=0.05,
            last_epoch=-1,
        )

        self.criterion = nn.CrossEntropyLoss().to(self.device)

    
    def train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        self.model.zero_grad()

        cum_loss = 0.0
        proc_loss = 0.0
        accuracy = 0
        proc_accuracy = 0.0

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                    desc=f'Train {self.current_epoch}/{self.hyp["epochs"] - 1}')
        for idx, (xi, xj, label, img) in pbar:
            xi, label = xi.to(self.device), label.to(self.device)

            with torch.set_grad_enabled(True):
                out = self.model(xi)
                loss = self.criterion(xi, label)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.model.zero_grad()

            cur_loss = loss.detach().cpu().numpy()
            cum_loss += cur_loss
            proc_loss = (proc_loss * idx + cur_loss) / (idx + 1)

            _, pred = torch.softmax(out.detach(), dim=1).topk(k=1)
            cur_accuracy = accuracy_score(label.detach().cpu(), pred.detach().cpu())
            accuracy += cur_accuracy
            proc_accuracy = (proc_accuracy * idx + cur_accuracy) / (idx + 1)

            s = f'Train {self.current_epoch}/{self.hyp["epochs"] - 1}, Loss: {proc_loss:4.3f}, Acc: {proc_accuracy:4.3f}'
            pbar.set_description(s)

        cum_loss /= len(self.train_loader)
        accuracy = accuracy / len(self.train_loader)

        return [cum_loss, accuracy]

    def valid_step(self):
        
        self.model.eval()

        cum_loss = 0.0
        proc_loss = 0.0

        accuracy = 0
        proc_accuracy = 0.0

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader),
                    desc=f'Valid {self.current_epoch}/{self.hyp["epochs"] - 1}')
        for idx, (xi, xj, label, img) in pbar:
            xi, label = xi.to(self.device), label.to(self.device)

            with torch.set_grad_enabled(False):
                out = self.model(xi)
                loss = self.criterion(xi, label)

            cur_loss = loss.detach().cpu().numpy()
            cum_loss += cur_loss
            proc_loss = (proc_loss * idx + cur_loss) / (idx + 1)

            _, pred = torch.softmax(out.detach(), dim=1).topk(k=1)
            cur_accuracy = accuracy_score(label.detach().cpu(), pred.detach().cpu())
            accuracy += cur_accuracy
            proc_accuracy = (proc_accuracy * idx + cur_accuracy) / (idx + 1)

            s = f'Valid {self.current_epoch}/{self.hyp["epochs"] - 1}, Loss: {proc_loss:4.3f}, Acc: {proc_accuracy:4.3f}'
            pbar.set_description(s)

        cum_loss /= len(self.valid_loader)
        accuracy /= len(self.valid_loader)
        return [cum_loss, accuracy]
    
    def run(self):

        train_losses = []
        valid_losses = []

        for epoch in range(self.hyp['epochs']):
            self.current_epoch = epoch

            loss_train = self.train_step()
            train_losses.append(loss_train)

            if epoch < 10:
                self.warmupscheduler.step()
            else:
                self.mainscheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]

            loss_valid = self.valid_step()
            valid_losses.append(loss_valid)

        torch.cuda.empty_cache()

        return train_losses, valid_losses

def main():
    datapath = "images_background"
    with open('hw4/src/SimCLR/hyp_params.yaml', 'r') as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
 

    train_dataset, valid_dataset = get_datasets(datapath)
    train_loader = DataLoader(train_dataset,
                                    batch_size=hyp['batch_size'],
                                    shuffle=True,
                                    num_workers=hyp['n_workers'],
                                    pin_memory=True,
                                    drop_last=True
                              )
    
    valid_loader = DataLoader(valid_dataset,
                                    batch_size=hyp['batch_size'],
                                    shuffle=True,
                                    num_workers=hyp['n_workers'],
                                    pin_memory=True,
                                    drop_last=True
                              )
    

    trainer = BaseTrainProcess(hyp, train_loader, valid_loader)
    train_losses, valid_losses = trainer.run()

    
    classifier = ClassifierCLR(trainer.model.encoder, trainer.model.emb_size)
    classifier_trainer = ClassifierCLRTrainer(classifier, hyp, train_loader, valid_loader)
    train_losses, valid_losses = classifier_trainer.run()

    model = torchvision.models.resnet50(pretrained=True)
    encoder = nn.Sequential(*tuple(model.children())[:-1])
    classifier_without_clr = ClassifierCLR(encoder, trainer.model.emb_size)
    classifier_without_clr.make_requires_grad(True)
    classifier_trainer = ClassifierCLRTrainer(classifier_without_clr, hyp, train_loader, valid_loader)
    train_losses_without, valid_losses_without = classifier_trainer.run()


if __name__ == "__main__":
    main()