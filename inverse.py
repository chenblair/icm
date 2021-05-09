import numpy as np
import torch

def find_inverse(experts, data, metric_func=torch.nn.MSELoss(), right_inverse=True):
    n_experts = len(experts)
    scores = np.zeros((n_experts, n_experts))

    with torch.no_grad:
        for batch in data:
            for idx, expert in enumerate(experts):
                transformed = expert(batch)
                for idx2, expert2 in enumerate(experts):
                    transformed2 = expert2(transformed)
                    scores[idx, idx2] += metric_func(batch, transformed2)
    
    if (right_inverse):
        return np.argmax(scores, axis=1)
    return np.argmax(scores, axis=0)
