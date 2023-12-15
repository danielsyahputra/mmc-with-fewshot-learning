from __future__ import print_function

import os
import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=os.path.dirname(os.path.realpath('__file__')),
    indicator=["requirements.txt"],
    pythonpath=True,
    dotenv=True,
)
import hydra
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import grad
import time
from tqdm.auto import tqdm
from omegaconf import DictConfig
from torch import autograd
from PIL import ImageFile
import sys
from timeit import default_timer as timer
sys.dont_write_bytecode = True

# Internal package
from src.utils import AverageMeter, accuracy, adjust_learning_rate, save_checkpoint
from src.dataset import FewShotDataset, get_dataloader
from src.models.network import define_model

def train(cfg, loader, model, criterion, optimizer, epoch_index, device=torch.device("cpu")):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()
    for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(loader):
        data_time.update(time.time() - end)

        try:
            input_var1 = torch.cat(query_images, 0).to(device)
        except:
            print(query_images)
        input_var2 = torch.cat(support_images, 0).squeeze(0).to(device)
        input_var2 = input_var2.contiguous().view(-1, input_var2.size(2), input_var2.size(3), input_var2.size(4))

        target = torch.cat(query_targets, 0).to(device)

        output = model(input_var1, input_var2)
        loss = criterion(output, target)

        prec1, _ = accuracy(output, target, topk=(1,3))
        losses.update(loss.item(), target.size(0))
        top1.update(prec1[0], target.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        #============== print the intermediate results ==============#

        if episode_index % cfg.print_freq == 0 and episode_index != 0:
            print('Epoch-({0}): [{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch_index, episode_index, len(loader), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1))
    
    return top1.avg, losses.avg

@torch.inference_mode()
def validate(cfg, loader, model, criterion, epoch_index, best_prec1, device=torch.device("cpu")):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    accuracies = []
    end = time.time()
    for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(loader):
        # Convert query and support images
        input_var1 = torch.cat(query_images, 0).to(device)
        input_var2 = torch.cat(support_images, 0).squeeze(0).to(device)
        input_var2 = input_var2.contiguous().view(-1, input_var2.size(2), input_var2.size(3), input_var2.size(4))


        # Deal with the targets
        target = torch.cat(query_targets, 0).to(device)

        output = model(input_var1, input_var2)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        prec1, _ = accuracy(output, target, topk=(1,3))
        losses.update(loss.item(), target.size(0))
        top1.update(prec1[0], target.size(0))
        accuracies.append(prec1)


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if episode_index % cfg.print_freq == 0 and episode_index != 0:
            print('Val-({0}): [{1}/{2}]\t'
                'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                'Loss {loss.val:.2f} ({loss.avg:.2f})\t'
                'Prec@1 {top1.val:.2f} ({top1.avg:.2f})'.format(
                    epoch_index, episode_index, len(loader), batch_time=batch_time, loss=losses, top1=top1))
    
    print(' * Prec@1 {top1.avg:.2f} Best_prec1 {best_prec1:.2f}'.format(top1=top1, best_prec1=best_prec1))
    return top1.avg, losses.avg

@hydra.main(config_path=f"{ROOT}/configs", config_name="main", version_base=None)
def main(cfg: DictConfig):
    cfg = cfg.train
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.cuda) else "cpu")
    best_prec1 = 0
    model = define_model(encoder_model=cfg.encoder_model, classifier_model=cfg.classifier_model, norm="batch",
                         way_num=cfg.way_num, shot_num=cfg.shot_num, init_type='normal', use_gpu=cfg.cuda)
    # model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    if cfg.adam:
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.9), weight_decay=0.0005)
        print("Using Adam Optimizer")
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, dampening=0.9, weight_decay=0.001)
        print("Using SGD Optimizer")
    
    if cfg.ngpu > 1:
        model = nn.DataParallel(model, range(cfg.ngpu))

    if cfg.cosine:
        eta_min = cfg.lr * (cfg.lr_decay_rate ** 3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs, eta_min, -1)
    
    # ============================================ Training phase ========================================
    print('===================================== Training on the train set =====================================')
    try:
        experiment_id = mlflow.create_experiment(cfg.experiment_name)
    except:
        current_experiment = dict(mlflow.get_experiment_by_name(cfg.experiment_name))
        experiment_id = current_experiment['experiment_id']
    
    with mlflow.start_run(experiment_id=experiment_id, run_name=cfg.run_name):
        start_time = timer()
        for epoch in tqdm(range(cfg.start_epoch, cfg.epochs + 1)):
            print('==================== Epoch %d ====================' %epoch)
            cfg.current_epoch = epoch
            train_loader, val_loader = get_dataloader(opt=cfg, modes=["train", "val"])
            prec1_train, train_loss = train(cfg, train_loader, model, criterion, optimizer, epoch, device=device)

            print('===================================== Validation on the val set =====================================')
            prec1_val, val_loss = validate(cfg, val_loader, model, criterion, epoch, best_prec1, device)

            # Adjust the learning rates
            if cfg.cosine:
                scheduler.step()
            else:
                adjust_learning_rate(cfg, optimizer, epoch)
            
            # remember best prec@1 and save checkpoint
            is_best = prec1_val > best_prec1
            best_prec1 = max(prec1_val, best_prec1)


            if is_best:
                save_path = f"{ROOT}/{cfg.outf}"
                os.makedirs(save_path, exist_ok=True)
                save_checkpoint({
                        'epoch_index': epoch,
                        'encoder_model': cfg.encoder_model,
                        'classifier_model': cfg.classifier_model,
                        'model': model.state_dict(),
                        'best_prec1': best_prec1,
                        'optimizer' : optimizer.state_dict(),
                    }, save_path)
                
            mlflow.log_metrics({
                "train_prec1": float(prec1_train),
                "train_loss": float(train_loss),
                "val_prec1": float(prec1_val),
                "val_loss": float(val_loss)
            }, step=epoch)
        
        mlflow.log_params(cfg)
        
        end_time = timer()
        mlflow.log_metrics({"time": end_time - start_time})
        mlflow.end_run()

    return model

if __name__=="__main__":
    main()