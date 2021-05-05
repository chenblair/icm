"""Main training script."""
import argparse
import json

import torch
import numpy as np

import expert as exper
import discriminator as discrim
from data import translated_gaussian_dataset, transformed_mnist_dataset
import metrics
from train_utils import initialize_experts
from train_utils import train_icm

from tqdm import tqdm 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICM experiment configuration")
    parser.add_argument(
        "--num_experts",
        type=int,
        default=8,
        help="number of experts (default: 5)",
    )
    parser.add_argument(
        "--input_shape",
        default=2,
        help="Size of the input shape",
    )
    parser.add_argument(
        "--discriminator_output_size",
        default=1,
        help="Size of the discriminator output",
    )
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=20,
        help="number of icm training epoch (default: 20)",
    )
    parser.add_argument(
        "--min_initialization_loss",
        type=float,
        default=0.01,
        help="Minimum loss before initialization is terminated",
    )
    parser.add_argument(
        "--num_initialize_epoch",
        type=int,
        default=10,
        help="Number of epochs the data is passed over",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Size of the minibatch",
    )

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    args.batch_size = 256
    args.num_initialize_epoch = 20
    args.min_initialization_loss = 0.1
    args.num_epoch = 0
    args.discriminator_output_size = 1
    # args.input_shape = 2
    args.input_shape = [28, 28, 1]
    args.use_sn = True
    args.num_experts = 5
    args.discriminator_sigmoid = False
    args.noise_scale = 0.1
    args.print_iterval = 200
    args.no_source_target = False
    args.num_transform = 1
    args.width_multiplier = 1
    args.load_experts_init = True
    args.load_experts = True

    args.experts_init_file = 'saved_models/expert_init_transforms_1.pt'
    args.experts_file = 'saved_models/base_icm_transforms_1.pt'
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # Data
    data = transformed_mnist_dataset(args.batch_size, args, test=True)
    # data = translated_gaussian_dataset(args.batch_size, args)

    # Model
    experts = [exper.ConvolutionExpert(args).to(args.device) for i in range(args.num_experts)]
    discriminator = discrim.ConvolutionDiscriminator(args).to(args.device)

    if (args.load_experts_init):
        init_experts = torch.load(args.experts_init_file, map_location=args.device)['expert_state_dicts']
        for idx, e in enumerate(experts):
            e.load_state_dict(init_experts[idx])
    else:
        initialize_experts(experts, data, args)
        torch.save( {
            'expert_state_dicts': [e.state_dict() for e in experts]
        }, args.experts_init_file)

    discriminator_opt = torch.optim.Adam(discriminator.parameters())
    expert_opt = []
    for e in experts:
        expert_opt.append(torch.optim.Adam(e.parameters()))

    if (args.load_experts):
        checkpoint = torch.load(args.experts_file, map_location=args.device)
        for idx, e in enumerate(experts):
            e.load_state_dict(checkpoint['expert_state_dicts'][idx])
        for idx, e in enumerate(expert_opt):
            e.load_state_dict(checkpoint['expert_optimizer_state_dicts'][idx])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        discriminator_opt.load_state_dict(checkpoint['discriminator_optimizer_state_dict'][0])

    for n in tqdm(range(args.num_epoch)):
        train_icm(experts, expert_opt, discriminator, discriminator_opt, data, args)
        torch.save( {
            'expert_optimizer_state_dicts': [e.state_dict() for e in expert_opt],
            'expert_state_dicts': [e.state_dict() for e in experts],
            'discriminator_optimizer_state_dict': [discriminator_opt.state_dict()],
            'discriminator_state_dict': discriminator.state_dict()
        }, args.experts_file)
    print('SSIM', metrics.get_ssim(experts, discriminator, data, args))
    
    
