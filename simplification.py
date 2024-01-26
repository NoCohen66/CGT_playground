# python certify.py -d 'mnist' -c 1 -p 'networks/mnist/cgt_rotate-30^30^0.5.pt' -b 3


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
from torchvision.utils import save_image
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


from configs.mnist_config import *

# LOAD NETWORK AND GET INDICES OF CORRECTLY CLASSIFIED IMAGES
dataset = get_dataset(get_train=False)
print('Number of images:', len(dataset), file=sys.stderr)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
net = Network().to(device)
net.load_state_dict(torch.load(network_path, map_location=torch.device('cpu')))
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
correct_dataloader = torch.utils.data.DataLoader(correct_imgs, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# CERTIFY ROBUSTNESS VIA INTERVAL BOUNDS
ibp_net = BoundedModule(net, torch.empty((batch_size, *img_shape)))
robust_indices = []
print('Certifying...', file=sys.stderr)

s_time = timer()



pixelwise_transforms = [('brightness', -0.01, 0.01)]
# start certification
with torch.no_grad():
    for batch_i, (inputs, labels) in enumerate(correct_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        robust_bool = torch.full((len(inputs),), True, device=device)  # stores which images of current batch are robust
        interval_inputs = IntervalTensor(inputs, inputs)
        # REPLACE LINE WITH THE TWO FOLLOWINGS
        # interval_inputs = pixelwise(interval_inputs, pixelwise_transforms) 
        transform, theta_min, theta_max = pixelwise_transforms[0]
        interval_inputs = IntervalTensor((interval_inputs.lower + theta_min).clamp(0, 1), (interval_inputs.upper + theta_max).clamp(0, 1))
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






