import torch
from datetime import datetime


def checkpoint(model, optimizer, base_filename="model"):
    """
    Save a PyTorch model and optimizer to a file with a name that includes the current timestamp.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        base_filename (str): The base name for the saved file (default is "model").
    """
    # Get the current timestamp in the format YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct the full filename with timestamp
    filename = f"checkpoints/{base_filename}_{timestamp}.pth"

    # Save the model's and optimizer's state dictionaries
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filename,
    )

    print(f"Model and optimizer saved as {filename}")
