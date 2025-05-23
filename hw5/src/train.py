import torch
from tqdm import tqdm
from torch import nn

from sklearn.metrics import accuracy_score

from model import SiameseNetwork
from params import DEVICE, hyp
from data import get_train_test_dataloader

import wandb

class Trainer:
    def __init__(self, model, criterion, hyp, train_loader, valid_loader):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = model.to(self.device)

        self.hyp = hyp
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self._init_model()

        self.criterion = criterion

    
    def _init_model(self):
        model_params = [params for params in self.model.parameters() if params.requires_grad]
        self.optimizer = torch.optim.AdamW(model_params, lr=self.hyp['lr'], weight_decay=self.hyp['weight_decay'])

    
    def train_step(self):
        self.optimizer.zero_grad()
        self.model.zero_grad()

        cumulative_loss = 0
        proc_loss = 0
        accuracy = 0
        proc_accuracy = 0 
        
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                    desc=f'Train {self.current_epoch}/{self.hyp["epochs"] - 1}')
        for idx, (x1, x2, label) in pbar:
            x1, x2, label = x1.to(self.device), x2.to(self.device), label.float().to(self.device)
            
            with torch.set_grad_enabled(True):
                pred = self.model(x1, x2)
                loss = self.criterion(pred.squeeze(), label)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.model.zero_grad()

            cur_loss = loss.detach().cpu().numpy()
            cumulative_loss += cur_loss
            proc_loss = (proc_loss * idx + cur_loss) / (idx + 1)

            pred_classes = (pred > 0.5).long()
            cur_accuracy = accuracy_score(label.cpu().numpy(), pred_classes.cpu().numpy())
            accuracy += cur_accuracy
            proc_accuracy = (proc_accuracy * idx + cur_accuracy) / (idx + 1)

            s = f'Train {self.current_epoch}/{self.hyp["epochs"] - 1}, Loss: {proc_loss:4.3f}, Acc: {proc_accuracy:4.3f}'
            pbar.set_description(s)

        cumulative_loss /= len(self.train_loader)
        accuracy /= len(self.train_loader)

        wandb.log({
            "train/loss": cumulative_loss,
            "train/accuracy": accuracy
        })

        return [cumulative_loss, accuracy]

    def valid_step(self):
        self.model.eval()

        cumulative_loss = 0
        proc_loss = 0
        accuracy = 0
        proc_accuracy = 0 

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader),
                    desc=f'Valid {self.current_epoch}/{self.hyp["epochs"] - 1}')
        for idx, (x1, x2, labels) in pbar:
            x1, x2, labels = x1.to(self.device), x2.to(self.device), labels.float().to(self.device)

            with torch.no_grad():
                pred = self.model(x1, x2)
                loss = self.criterion(pred.squeeze(), labels)

            cur_loss = loss.detach().cpu().numpy()
            cumulative_loss += cur_loss
            proc_loss = (proc_loss * idx + cur_loss) / (idx + 1)

            pred_classes = (pred > 0.5).long()
            cur_accuracy = accuracy_score(labels.cpu().numpy(), pred_classes.cpu().numpy())
            accuracy += cur_accuracy
            proc_accuracy = (proc_accuracy * idx + cur_accuracy) / (idx + 1)

            s = f'Valid {self.current_epoch}/{self.hyp["epochs"] - 1}, Loss: {proc_loss:4.3f}, Acc: {proc_accuracy:4.3f}'
            pbar.set_description(s)
            
        cumulative_loss /= len(self.valid_loader)
        accuracy /= len(self.valid_loader)

        wandb.log({
            "valid/loss": cumulative_loss,
            "valid/accuracy": accuracy
        })

        return [cumulative_loss, accuracy]
    
    def run(self):

        train_losses = []
        valid_losses = []

        for epoch in range(self.hyp['epochs']):
            self.current_epoch = epoch

            loss_train = self.train_step()
            train_losses.append(loss_train)

            loss_valid = self.valid_step()
            valid_losses.append(loss_valid)

        torch.cuda.empty_cache()

        return train_losses, valid_losses
    

def main():
    train_dataloader, valid_dataloader = get_train_test_dataloader()

    model = SiameseNetwork().to(DEVICE)
    criterion = nn.BCELoss().to(DEVICE)
    trainer = Trainer(model, criterion, hyp, train_dataloader, valid_dataloader)

    wandb.init(project=hyp.get("hw5", "siamese_training"), config=hyp)
    train, valid = trainer.run()

    torch.save(model.state_dict(), 'siamese_model.pt')

if __name__ == "__main__":
    main()