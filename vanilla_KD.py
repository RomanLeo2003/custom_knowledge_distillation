import torch
import torchvision
from accelerate import Accelerator
import random
import numpy as np
import os
import tqdm

def seed(seed_value):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

def vanilla_distillation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    teacher_scores: torch.Tensor,
    T: float,
    alpha: float
):
    return nn.KLDivLoss()(F.log_softmax(pred/T), F.softmax(teacher_scores/T)) * (T*T * 2.0 * alpha) + F.cross_entropy(pred, labels) * (1. - alpha)


def train_epoch(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    optimizer: torch.optim,
    criterion: torch.nn.modules.loss._Loss,
    loader: torch.utils.data.DataLoader,
    device: torch.device
):
    acc_loss = 0
    total = len(loader.dataset)
    student_model.train()
    teacher_model.eval()
    for data, target in tqdm.tqdm(loader):
      with accelerator.accumulate(model):  # для имитации большого размера батча (полезно для трансформеров)
        pred = torch.nn.functional.softmax(model(data), dim=-1)
        teacher_output = teacher_model(data)
        teacher_output = teacher_output.detach()
        loss = vanilla_distillation_loss(pred, target, teacher_output, T=20.0, alpha=0.7)
        # scaler.scale(loss).backward()
        # scaler.unscale_(optimizer)
        # scaler.step(optimizer)
        # loss.backward()
        accelerator.backward(loss) # вместо loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        acc_loss += loss.item()

    return acc_loss / total

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


EvalOut = namedtuple("EvalOut", ['loss', 'MSE'])

def eval_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    loader: torch.utils.data.DataLoader,
    device: torch.device
):
    mses = []
    total = len(loader.dataset)
    acc_loss = 0
    model.eval()
    # model.to(device)
    with torch.inference_mode():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            pred = model(data)
            loss = criterion(pred, target)
            acc_loss += loss.item()
            mses.append((pred - target) ** 2)

    return EvalOut(loss=(acc_loss / total), MSE=(sum(mses) / total))


class TrainOut(NamedTuple):
    train_loss: List[float]
    eval_loss: List[float]
    eval_accuracy: List[float]


def train(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    optimizer: torch.optim,
    criterion: torch.nn.modules.loss._Loss,
    sheduler: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 10
):
    train_loss = []
    eval_loss = []
    eval_MSE = []
    model.to(device)
    for i in range(epochs):
        print(f"Epoch - {i}:")
        if (train_loader != None):
            print("Train...\n")
            train_loss.append(train_epoch(student_model, teacher_model, optimizer, criterion, train_loader, device))
        print("Validation...\n")
        eval_out = eval_epoch(student_model, criterion, val_loader, device)
        eval_loss.append(eval_out.loss)
        eval_MSE.append(eval_out.MSE)
        print(f'Validation MSE: {eval_out.MSE}')
        sheduler.step()
        print('lr: ', get_lr(optimizer))
        if i > 1 and eval_MSE[i] == min(eval_MSE):
          unwrapped_model = accelerator.unwrap_model(model)
          accelerator.save({
                "model": model.state_dict(),
                "optimizer": optimizer.optimizer.state_dict() # optimizer is an AcceleratedOptimizer object
            }, "bundle.pth")
          torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoint_{i}.pth')

    return TrainOut(
        train_loss=train_loss,
        eval_loss=eval_loss,
        eval_accuracy=eval_MSE
    )

