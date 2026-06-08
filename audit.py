import torch.nn as nn

def check_parameter_count(model: nn.Module):
    total_trainable = 0
    total_frozen = 0

    for name, param in model.named_parameters():
        num_params = param.numel()

        if param.requires_grad:
            total_trainable += num_params
            print(f"[TRAINABLE] {name:40s} {tuple(param.shape)}")
        else:
            total_frozen += num_params
            print(f"[FROZEN]    {name:40s} {tuple(param.shape)}")

    print("\nSummary:")
    print("Trainable params:", total_trainable)
    print("Frozen params:", total_frozen)
    print("Total params:", total_trainable + total_frozen)