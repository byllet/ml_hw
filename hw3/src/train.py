from tqdm.auto import tqdm
import torch
import math
import wandb

from utilities import convert_batch
from params import config

def do_epoch(model, criterion, data_iter, epoch, optimizer=None, name=None):
    epoch_loss = 0
    
    is_train = not optimizer is None
    name = name or ''
    model.train(is_train)
    
    batches_count = len(data_iter)
    
    with torch.autograd.set_grad_enabled(is_train):
        with tqdm(total=batches_count) as progress_bar:
            for i, batch in enumerate(data_iter):
                source_inputs, target_inputs, source_mask, target_mask = convert_batch(batch)                                
                logits = model.forward(source_inputs, target_inputs[:, :-1], source_mask, target_mask[:, :-1, :-1])
                
                logits = logits.contiguous().view(-1, logits.shape[-1])
                target = target_inputs[:, 1:].contiguous().view(-1)
                loss = criterion(logits, target)

                epoch_loss += loss.item()

                if optimizer:
                    optimizer.optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                progress_bar.update()
                progress_bar.set_description('{:>5s} Loss = {:.5f}, PPX = {:.2f}'.format(name, loss.item(), 
                                                                                         math.exp(loss.item())))
                if i % 100:
                    torch.save(model.state_dict(), "models/model_temp.pt")
                if epoch_loss < 0.1 * i:
                    torch.save(model.state_dict(), "models/model_temp.pt")
                    break
                if is_train:
                    wandb.log({"loss": loss.item()}, step=i + epoch * batches_count)
                
            progress_bar.set_description('{:>5s} Loss = {:.5f}, PPX = {:.2f}'.format(
                name, epoch_loss / batches_count, math.exp(epoch_loss / batches_count))
            )
            progress_bar.refresh()
    torch.save(model.state_dict(), f"models/model_epoch{epoch}.pt")
    return epoch_loss / batches_count


def fit(model, criterion, optimizer, train_iter, start_epoch=0, epochs_count=30, val_iter=None):
    wandb.init(config=config, project="hw3", name="baseline")

    best_val_loss = None
    step = 0
    for epoch in range(start_epoch, epochs_count):
        name_prefix = '[{} / {}] '.format(epoch + 1, epochs_count)
        train_loss = do_epoch(model, criterion, train_iter, epoch + start_epoch, 
                              optimizer, name_prefix + 'Train:')

        metrics = {'train_loss': train_loss}
        step += len(train_iter)
        if val_iter is not  None:
            val_loss = do_epoch(model, criterion, val_iter, epoch + start_epoch, None, name_prefix + '  Val:')
            metrics['val_loss'] = val_loss
            if best_val_loss is None:
                best_val_loss = val_loss
            best_val_loss = min(best_val_loss, val_loss)
            step += len(val_iter)

        wandb.log(metrics, step=step)

    with open("run_id.txt", "w+") as f:
        print(wandb.run.id, file=f)
    with open("best_val_loss.txt", "w+") as f:
        print(best_val_loss, file=f)
