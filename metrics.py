import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

import pdb

def get_mse(experts, discriminator, data, args):
    """Execute the ICM training.

    Args:
        experts (Expert): a list of all experts
        expert_opt (list): a list of all optmizers for each expert
        discriminator (Discriminator): the discriminator
        discriminator_opt : the optimizer for training the discriminator
        data (DataLoader): pytorch dataloader for the training data
        args : argparse object
    """
    mse_loss = torch.nn.MSELoss(reduction="mean")
    loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
    discriminator.eval()
    _ = [e.eval() for e in experts]
    
    mse_losses = []
    with torch.no_grad():
        for idx, batch in enumerate(data):
            x_src, x_tgt = batch
            x_src, x_tgt = x_src.to(args.device), x_tgt.to(args.device)

            if args.no_source_target:
                x_src = x_tgt

            # D expert pass

            exp_out, exp_score = [], []
            loss_exp_d = 0
            labels = torch.full((args.batch_size,), 0.0, device=args.device).unsqueeze(dim=1)
            for e in experts:
                out = e(x_tgt)
                score = discriminator(out.detach())
                exp_out.append(out)
                exp_score.append(score)
                loss_exp_d += loss(score, labels)
            loss_exp_d /= len(experts)
            # loss_exp_d.backward()

            # D discriminator pass
            score = discriminator(x_src)
            # labels.fill_(1)
            labels = torch.full((args.batch_size,), 1.0, device=args.device).unsqueeze(dim=1)
            total_loss = loss(score, labels.detach()) + loss_exp_d

            exp_out = [torch.unsqueeze(out, 1) for out in exp_out]
            exp_out = torch.cat(exp_out, dim=1)
            exp_score = torch.cat(exp_score, dim=1)
            winning_idx = exp_score.argmax(dim=1)

            per_expert_winning_num = []

            for i, e in enumerate(experts):
                selected_idx = winning_idx.eq(i).nonzero().squeeze(dim=-1)
                n_samples = selected_idx.size(0)
                per_expert_winning_num.append(n_samples)
                if n_samples > 0:
                    # samples = exp_out[selected_idx, i]
                    samples = e(x_tgt[selected_idx])
                    mse_losses.append(mse_loss(samples, x_src[selected_idx]).cpu().item())
    return np.mean(mse_losses)

def get_ssim(experts, discriminator, data, args):
    """Execute the ICM training.

    Args:
        experts (Expert): a list of all experts
        expert_opt (list): a list of all optmizers for each expert
        discriminator (Discriminator): the discriminator
        discriminator_opt : the optimizer for training the discriminator
        data (DataLoader): pytorch dataloader for the training data
        args : argparse object
    """
    loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
    discriminator.eval()
    _ = [e.eval() for e in experts]
    
    ssim_losses = []
    with torch.no_grad():
        for idx, batch in enumerate(data):
            x_src, x_tgt = batch
            x_src, x_tgt = x_src.to(args.device), x_tgt.to(args.device)

            if args.no_source_target:
                x_src = x_tgt

            # D expert pass

            exp_out, exp_score = [], []
            loss_exp_d = 0
            labels = torch.full((args.batch_size,), 0.0, device=args.device).unsqueeze(dim=1)
            for e in experts:
                out = e(x_tgt)
                score = discriminator(out.detach())
                exp_out.append(out)
                exp_score.append(score)
                loss_exp_d += loss(score, labels)
            loss_exp_d /= len(experts)
            # loss_exp_d.backward()

            # D discriminator pass
            score = discriminator(x_src)
            # labels.fill_(1)
            labels = torch.full((args.batch_size,), 1.0, device=args.device).unsqueeze(dim=1)
            total_loss = loss(score, labels.detach()) + loss_exp_d

            exp_out = [torch.unsqueeze(out, 1) for out in exp_out]
            exp_out = torch.cat(exp_out, dim=1)
            exp_score = torch.cat(exp_score, dim=1)
            winning_idx = exp_score.argmax(dim=1)

            per_expert_winning_num = []

            for i, e in enumerate(experts):
                selected_idx = winning_idx.eq(i).nonzero().squeeze(dim=-1)
                n_samples = selected_idx.size(0)
                per_expert_winning_num.append(n_samples)
                if n_samples > 0:
                    # print('Expert', i)
                    # samples = exp_out[selected_idx, i]
                    samples = e(x_tgt[selected_idx])
                    for sample, s_idx in zip(samples, selected_idx):
                        sample_np = sample[0].cpu().numpy()
                        src_np = x_src[s_idx,0].cpu().numpy()
                        # f, axarr = plt.subplots(nrows=1,ncols=3)
                        # axarr[0].axis('off')
                        # plt.sca(axarr[0]); 
                        # plt.imshow(x_tgt[s_idx, 0].cpu().numpy()); plt.title('Transformed')
                        # plt.sca(axarr[1]); 
                        # axarr[1].axis('off')
                        # plt.imshow(sample_np); plt.title('Expert')
                        # plt.sca(axarr[2]); 
                        # axarr[2].axis('off')
                        # plt.imshow(src_np); plt.title('Original')
                        # plt.show()
                        # ssim_losses.append(ssim(sample_np, src_np, data_range=1.0))
                        # plt.imshow(src_np)
                        # plt.savefig('image1.png')
                        # plt.clf()
                        # pdb.set_trace()
                        ssim_losses.append(ssim(sample_np, src_np, data_range=1.0))
    return np.mean(ssim_losses)
    