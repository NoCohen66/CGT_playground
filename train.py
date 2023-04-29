import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as Ft
import torch.optim as optim

import numpy as np
import os
import random
import sys
from timeit import default_timer as timer

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm, BoundDataParallel
from torchattacks import PGD

from geometric.intervals import *
from geometric.interval_perturbations import *
from geometric.scalar_perturbations import *

import argparse
parser = argparse.ArgumentParser(description='Train robust networks against geometric transformations')
parser.add_argument('-d', '--dataset-name', choices=[MNIST, CIFAR10, TINY, DRIVING])
parser.add_argument('-t', '--train-mode', default=CGT_IBP, choices=[AUGMENTED, PGD_A, IBP_A, CGT_IBP])
parser.add_argument('-c', '--config-num', type=int)
parser.add_argument('-b', '--batch-size', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--epsilon', type=float)           # param only used in l-infinity methods
parser.add_argument('--save-every', type=int)          # save model every few epochs
parser.add_argument('--validate-every', type=int)
parser.add_argument('--save-dir')                      # where to save models
parser.add_argument('--seed', default=2022, type=int)
parser.add_argument('--dropout', action='store_true')  # whether to use dropout for self-driving network
parser.add_argument('--multi-gpu', action='store_true')
args = parser.parse_args()


# SEED THINGS
seed = args.seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(seed)
print(f'Seed set to {seed}', file=sys.stderr)


# PARAMS
dataset_name = args.dataset_name
train_mode = args.train_mode
batch_size = args.batch_size
num_epochs = args.epochs
lr = args.lr
linf_eps = args.epsilon


# TRAINING HYPERPARAMETERS
if train_mode == CGT_IBP or train_mode == IBP_A:
    ramp_up_epochs = num_epochs / 2  # number of epochs used to linearly increase eps from 0 to specified interval size
    kappa_f = 0.5                    # final weighting of interval loss term
    if dataset_name == MNIST:
        start_interval_epoch = 16    # epoch at which to start using interval loss term
    elif dataset_name == CIFAR10:
        start_interval_epoch = 31
    elif dataset_name == TINY:
        start_interval_epoch = 31
    elif dataset_name == DRIVING:
        start_interval_epoch = 1
        ramp_up_epochs = num_epochs
elif train_mode == PGD_A:
    if dataset_name == MNIST:
        pgd_alpha = 0.005
        pgd_steps = 40
    elif dataset_name == CIFAR10:
        pgd_alpha = 0.002
        pgd_steps = 7


# LOAD DATA AND INITIALIZE NETWORK
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, file=sys.stderr)

if dataset_name == MNIST:
    from configs.mnist_config import *
elif dataset_name == CIFAR10:
    from configs.cifar10_config import *
elif dataset_name == TINY:
    from configs.tinyimagenet_config import *
elif dataset_name == DRIVING:
    from configs.driving_config import *

gts = gtss_train[args.config_num]
print('Transforms:', file=sys.stderr)
for tfm in gts:
    params = gts[tfm]
    if train_mode == CGT_IBP:
        print(f'{tfm}: [{params[0]}, {params[1]}], w={params[2]}', file=sys.stderr)
    else:
        print(f'{tfm}: [{params[0]}, {params[1]}]', file=sys.stderr)

if args.validate_every is not None:
    trainset, valset = get_dataset(get_val=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
else:
    trainset = get_dataset()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)

if not os.path.isdir(f'{args.save_dir}/{dataset_name}'):
    os.system(f'mkdir -p {args.save_dir}/{dataset_name}')
save_path = f'{args.save_dir}/{dataset_name}/{"cgt" if train_mode == CGT_IBP else train_mode}_'
for g, (pmin, pmax, interval_size) in gts.items():
    if train_mode == CGT_IBP:
        save_path += g + f'{pmin}^{pmax}^{interval_size}' + '_'
    else:
        save_path += g + f'{pmin}^{pmax}' + '_'
if train_mode == IBP_A or train_mode == PGD_A:
    save_path += f'eps{linf_eps}_'
save_path = save_path + f'seed{seed}.pt'

print(f'Network save path: {save_path}', file=sys.stderr)

if dataset_name == DRIVING and args.dropout:
    net = Network(dropout=True).to(device)
    print('Using dropout', file=sys.stderr)
else:
    net = Network().to(device)

optimizer = optim.Adam(net.parameters(), lr=lr)
if dataset_name == DRIVING:
    criterion = nn.MSELoss()
else:
    criterion = nn.CrossEntropyLoss()

# learning rate scheduler
if dataset_name == MNIST:
    milestones = [80]
elif dataset_name == CIFAR10:
    milestones = [100]
elif dataset_name == TINY:
    milestones = [120, 150]
elif dataset_name == DRIVING:
    milestones = [20, 40]
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

if 'ibp' in train_mode:
    ibp_net = BoundedModule(net, torch.empty((batch_size, *img_shape)))
    if args.multi_gpu:
        ibp_net = BoundDataParallel(ibp_net)
    kappa_step = (1 - kappa_f) / (num_epochs - start_interval_epoch + 1)
    kappa = 1

    if train_mode == CGT_IBP:
        eps_step = {}
        eps = {}
        for p in gts:
            eps_step[p] = gts[p][2] / ramp_up_epochs
            eps[p] = 0
    else:
        eps_step = linf_eps / ramp_up_epochs
        eps = 0
elif train_mode == PGD_A:
    atk = PGD(net, eps=linf_eps, alpha=pgd_alpha, steps=pgd_steps)


# START TRAINING
total_time = 0
print(f'Start training {dataset_name} network with mode: {"cgt" if train_mode == CGT_IBP else train_mode}', file=sys.stderr)

for epoch in range(num_epochs):
    print(f'**Epoch {epoch+1}**', file=sys.stderr)

    if args.save_every is not None and epoch % args.save_every == 0:
       torch.save(net.state_dict(), save_path[:-3] + f'_ep{epoch}.pt')

    torch.cuda.reset_peak_memory_stats(device=device)
    running_loss = 0.0

    if 'ibp' in train_mode and epoch + 1 >= start_interval_epoch:
        kappa -= kappa_step
        print(f'kappa: {kappa}', file=sys.stderr)

        if epoch + 1 < start_interval_epoch + ramp_up_epochs:
            if train_mode == CGT_IBP:
                for p in gts:
                    eps[p] += eps_step[p]
                    print(f'{p} eps: {eps[p]}', file=sys.stderr)
            else:
                eps += eps_step
                print(f'eps: {eps}', file=sys.stderr)
    
    start = timer()

    for (inputs, labels) in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # augment images with a transform that is sampled within the perturbation ranges specified
        # positive rotation in PyTorch is in clockwise direction, while ours is in counter-clockwise direction;
        # similarly, positive y-translation in PyTorch is downwards, while ours is upwards, hence negative signs below
        
        tv = {
            ROTATE: 0.0,
            TRANSLATE_X: 0.0,
            TRANSLATE_Y: 0.0,
            SCALE: 1.0,
            SHEAR: 0.0,

            CONTRAST: 1.0,
            BRIGHTNESS: 0.0,
        }

        if train_mode == CGT_IBP:
            interp_transforms = []
            pixelwise_transforms = []
        
        for (p_name, (pmin, pmax, _)) in gts.items():
            sampled_tfm_val = random.uniform(pmin, pmax)
            if p_name == ROTATE or p_name == TRANSLATE_Y:
                tv[p_name] = -sampled_tfm_val
            elif p_name == SHEAR:
                tv[p_name] = np.rad2deg(np.arctan(sampled_tfm_val))  # unit conversion from % to degrees of shear (for PyTorch affine)
            else:
                tv[p_name] = sampled_tfm_val

            if train_mode == CGT_IBP:
                # interval bound checking to make sure interval does not exceed specified min/max bounds
                if sampled_tfm_val - eps[p_name]/2 < pmin:
                    tfm_interval = (pmin, pmin + eps[p_name])
                elif sampled_tfm_val + eps[p_name]/2 > pmax:
                    tfm_interval = (pmax - eps[p_name], pmax)
                else:
                    tfm_interval = (sampled_tfm_val - eps[p_name]/2, sampled_tfm_val + eps[p_name]/2)

                if p_name in geometric_perturbations:
                    interp_transforms.append((p_name, tfm_interval[0], tfm_interval[1]))
                elif p_name in pixelwise_perturbations:
                    pixelwise_transforms.append((p_name, tfm_interval[0], tfm_interval[1]))
                else:
                    raise ValueError('No such perturbation implemented!')
        
        augmented_inputs = inputs
        if tv[ROTATE] != 0 or tv[TRANSLATE_X] != 0 or tv[TRANSLATE_Y] != 0 or tv[SCALE] != 1 or tv[SHEAR] != 0:
            augmented_inputs = Ft.affine(augmented_inputs, tv[ROTATE], (tv[TRANSLATE_X], tv[TRANSLATE_Y]), tv[SCALE], tv[SHEAR], interpolation=Ft.InterpolationMode.BILINEAR)
        if tv[CONTRAST] != 1 or tv[BRIGHTNESS] != 0:
            augmented_inputs = contrast_brightness(augmented_inputs, tv[CONTRAST], tv[BRIGHTNESS])

        if train_mode == AUGMENTED:
            outputs = net(augmented_inputs)
            loss = criterion(outputs, labels)
        elif 'ibp' in train_mode and epoch + 1 < start_interval_epoch:
            outputs = ibp_net(augmented_inputs)
            loss = criterion(outputs, labels)
        elif train_mode == PGD_A:
            adv_inputs = atk(augmented_inputs, labels)
            outputs = net(adv_inputs)
            loss = criterion(outputs, labels)
        elif 'ibp' in train_mode:
            if train_mode == IBP_A:
                ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
            else:
                interval_inputs = IntervalTensor(inputs, inputs)
                if len(interp_transforms) > 0:
                    interp_grid = make_interp_grid(img_shape, interp_transforms)
                    interval_inputs = spatial(interval_inputs, interp_grid)
                if len(pixelwise_transforms) > 0:
                    interval_inputs = pixelwise(interval_inputs, pixelwise_transforms)
                ptb = PerturbationLpNorm(norm=np.inf, x_L=interval_inputs.lower, x_U=interval_inputs.upper)

            outputs = ibp_net(augmented_inputs)
            bounded_inputs = BoundedTensor(augmented_inputs, ptb)  # note: for our training method, bounded_inputs will be directly set via x_L, x_U (i.e., does not depend on augmented_inputs)
            lb, ub = ibp_net(method_opt="compute_bounds", x=(bounded_inputs,), method="IBP")

            if dataset_name == DRIVING:
                loss = kappa * criterion(outputs, labels) + (1 - kappa) * ((criterion(lb, labels) + criterion(ub, labels)) / 2)
            else:
                labels_onehot = F.one_hot(labels, num_classes=num_classes)
                worst_case_outputs = lb * labels_onehot + ub * torch.logical_not(labels_onehot)
                loss = kappa * criterion(outputs, labels) + (1 - kappa) * criterion(worst_case_outputs, labels)
        else:
            raise ValueError('No such training method!')

        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 8)
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    end = timer()
    total_time += end - start

    max_memory = torch.cuda.max_memory_allocated(device=device)

    # run on val set
    val_statistic = None
    if args.validate_every is not None and (epoch + 1) % args.validate_every == 0:
        net.eval()
        correct = 0
        running_val_loss = 0
        with torch.no_grad():
            for (images, labels) in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = ibp_net(images)
                if dataset_name == DRIVING:
                    running_val_loss += criterion(outputs, labels).item() * images.size(0)
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
        if dataset_name == DRIVING:
            val_statistic = round(running_val_loss / len(valset), 6)
        else:
            val_statistic = round(100 * correct / len(valset), 6)
        net.train()

    epoch_loss = round(running_loss / len(trainset), 6)
    epoch_time = round(end - start, 6)

    # print info for current epoch
    if val_statistic is not None:
        print(f'Loss: {epoch_loss}, Val {"Loss" if dataset_name == DRIVING else "Accuracy"}: {val_statistic}, Max Memory: {max_memory}, Time: {epoch_time}', file=sys.stderr)
        print(f'{epoch+1},{epoch_time},{max_memory},{epoch_loss},{val_statistic}')
    else:
        print(f'Loss: {epoch_loss}, Max Memory: {max_memory}, Time: {epoch_time}', file=sys.stderr)
        print(f'{epoch+1},{epoch_time},{max_memory},{epoch_loss}')

    # step training parameters
    scheduler.step()
    if (epoch + 1) in milestones:
        print('lr:', scheduler.get_last_lr(), file=sys.stderr)

print(f'Finished training {dataset_name} network in {total_time} seconds', file=sys.stderr)
torch.save(net.state_dict(), save_path[:-3] + f'_final.pt')
