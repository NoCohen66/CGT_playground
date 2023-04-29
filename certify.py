import torch
import torch.nn.functional as F

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm, BoundDataParallel

import itertools
import numpy as np
import os
from timeit import default_timer as timer
from tqdm import tqdm
import sys
sys.path.append('../')

from geometric.intervals import *
from geometric.interval_perturbations import *
from geometric.scalar_perturbations import *

import argparse
parser = argparse.ArgumentParser(description='Certify robustness of networks against geometric transformations')
parser.add_argument('-b', '--batch-size', type=int, default=1024)
parser.add_argument('-d', '--dataset-name', choices=[MNIST, CIFAR10, TINY])
parser.add_argument('-p', '--network-path')
parser.add_argument('-c', '--config-num', type=int)
parser.add_argument('--save', action='store_true')
parser.add_argument('--val', action='store_true')
parser.add_argument('--seed', default=2022, type=int)
parser.add_argument('--cpu-store', action='store_true')  # store computed interpolation grids on the cpu
parser.add_argument('--np', action='store_true')         # no precomputation of interpolation grids
parser.add_argument('--multi-gpu', action='store_true')
args = parser.parse_args()

batch_size = args.batch_size
dataset_name = args.dataset_name
network_path = args.network_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Network:', network_path, file=sys.stderr)


# LOAD DATASET AND TRANSFORMS
if dataset_name == MNIST:
    from configs.mnist_config import *
elif dataset_name == CIFAR10:
    from configs.cifar10_config import *
elif dataset_name == TINY:
    from configs.tinyimagenet_config import *
else:
    raise ValueError('No such dataset!')

if args.val:
    random.seed(args.seed)
    _, dataset = get_dataset(get_val=True)
else:
    dataset = get_dataset(get_train=False)

print('Number of images:', len(dataset), file=sys.stderr)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

gts = gtss_certify[args.config_num]
gts_split = {}
for g, (theta_min, theta_max, interval_size) in gts.items():
    num_splits = round((theta_max - theta_min) / interval_size)
    intervals = []
    for k in range(num_splits):
        intervals.append((g, theta_min + k * interval_size, theta_min + (k+1) * interval_size))
    gts_split[g] = intervals
    print(f'{g}: {num_splits} splits', file=sys.stderr)
all_gts_splits = list(itertools.product(*gts_split.values()))  # stores all perturbation splits for whole specified range

print('Transforms:', file=sys.stderr)
has_geometric = False
has_pixelwise = False
for tfm in gts:
    if tfm in geometric_perturbations:
        has_geometric = True
    elif tfm in pixelwise_perturbations:
        has_pixelwise = True
    params = gts[tfm]
    print(f'{tfm}: [{params[0]}, {params[1]}], w={params[2]}', file=sys.stderr)


# LOAD NETWORK AND GET INDICES OF CORRECTLY CLASSIFIED IMAGES
net = Network().to(device)
net.load_state_dict(torch.load(network_path))
net.eval()

correct = torch.full((len(dataset),), False)

with torch.no_grad():
    for batch_i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)

        if inputs.size(0) < batch_size:
            correct[batch_i * batch_size : ] = (predicted == labels)
        else:
            correct[batch_i * batch_size : (batch_i + 1) * batch_size] = (predicted == labels)

num_correct = correct.sum().item()
print('Correct:', num_correct, file=sys.stderr)

correct_indices = correct.nonzero().flatten()
correct_imgs = torch.utils.data.Subset(dataset, correct_indices)
correct_dataloader = torch.utils.data.DataLoader(correct_imgs, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


# CERTIFY ROBUSTNESS VIA INTERVAL BOUNDS
ibp_net = BoundedModule(net, torch.empty((batch_size, *img_shape)))
if args.multi_gpu:
    ibp_net = BoundDataParallel(ibp_net)
robust_indices = []
print('Certifying...', file=sys.stderr)

s_time = timer()

# precompute interpolation grids and pixelwise tfms
if not args.np:
    grids = [None for k in range(len(all_gts_splits))]
    pwises = [None for k in range(len(all_gts_splits))]
    for split_i, tfms in enumerate(tqdm(all_gts_splits)):
        interp_transforms = []
        pixelwise_transforms = []
        for (p_name, pmin, pmax) in tfms:
            if p_name in geometric_perturbations:
                interp_transforms.append((p_name, pmin, pmax))
            elif p_name in pixelwise_perturbations:
                pixelwise_transforms.append((p_name, pmin, pmax))
            else:
                raise ValueError('No such perturbation implemented!')
        if has_geometric:
            if args.cpu_store:
                (nonzero_row_indices, nonzero_col_indices, nonzero_vals, nonzero_counts) = make_interp_grid(img_shape, interp_transforms)
                grids[split_i] = (nonzero_row_indices.cpu(), nonzero_col_indices.cpu(), IntervalTensor(nonzero_vals.lower.cpu(), nonzero_vals.upper.cpu()), nonzero_counts.cpu())
            else:
                grids[split_i] = make_interp_grid(img_shape, interp_transforms)
        if has_pixelwise:
            pwises[split_i] = pixelwise_transforms

# start certification
with torch.no_grad():
    for batch_i, (inputs, labels) in enumerate(correct_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        robust_bool = torch.full((len(inputs),), True, device=device)  # stores which images of current batch are robust

        for split_i, tfms in enumerate(tqdm(all_gts_splits)):
            if not torch.any(robust_bool):
                break
            
            interval_inputs = IntervalTensor(inputs, inputs)
            
            if args.np:
                interp_transforms = []
                pixelwise_transforms = []
                for (p_name, pmin, pmax) in tfms:
                    if p_name in geometric_perturbations:
                        interp_transforms.append((p_name, pmin, pmax))
                    elif p_name in pixelwise_perturbations:
                        pixelwise_transforms.append((p_name, pmin, pmax))
                    else:
                        raise ValueError('No such perturbation implemented!')

                if has_geometric:
                    interp_grid = make_interp_grid(img_shape, interp_transforms)
                    interval_inputs = spatial(interval_inputs, interp_grid)
                if has_pixelwise:
                    interval_inputs = pixelwise(interval_inputs, pixelwise_transforms)
            else:
                if has_geometric:
                    if args.cpu_store:
                        (nonzero_row_indices, nonzero_col_indices, nonzero_vals, nonzero_counts) = grids[split_i]
                        interp_grid = (nonzero_row_indices.cuda(), nonzero_col_indices.cuda(), IntervalTensor(nonzero_vals.lower.cuda(), nonzero_vals.upper.cuda()), nonzero_counts.cuda())
                    else:
                        interp_grid = grids[split_i]
                    interval_inputs = spatial(interval_inputs, interp_grid)
                if has_pixelwise:
                    pixelwise_transforms = pwises[split_i]
                    interval_inputs = pixelwise(interval_inputs, pixelwise_transforms)

            inputs_L = interval_inputs.lower
            inputs_U = interval_inputs.upper
            ptb = PerturbationLpNorm(norm=np.inf, x_L=inputs_L, x_U=inputs_U)
            bounded_inputs = BoundedTensor(inputs, ptb)  # note: inputs will not actually be used here; instead the bounds inputs_L and inputs_U will be used directly
            lb, ub = ibp_net(method_opt="compute_bounds", x=(bounded_inputs,), method="IBP")
            
            labels_onehot = F.one_hot(labels, num_classes=num_classes)
            interval_outputs = lb * labels_onehot + ub * torch.logical_not(labels_onehot)
            _, predicted = torch.max(interval_outputs.data, 1)
            robust_bool = torch.logical_and(robust_bool, (predicted == labels))
        
        # record the indices (wrt to whole dataset) of certified robust images
        if inputs.size(0) < batch_size:
            robust_indices += correct_indices[batch_i * batch_size : ][robust_bool].tolist()
        else:
            robust_indices += correct_indices[batch_i * batch_size : (batch_i + 1) * batch_size][robust_bool].tolist()

f_time = timer()
certify_time = f_time - s_time
num_robust = len(robust_indices)
print('Robust:', num_robust, file=sys.stderr)
print('Time:', certify_time, file=sys.stderr)

print(f'{network_path},{num_correct},{num_robust},{certify_time}')

if args.save:
    os.system(f'mkdir -p certified_indices/{dataset_name}')
    torch.save(robust_indices, f'certified_indices/{dataset_name}/indices_{network_path.split("/")[-1]}')
