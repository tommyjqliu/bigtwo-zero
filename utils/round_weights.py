import torch


def round_list(arr, precision=5):
    if isinstance(arr[0], list):
        return [round_list(sub_arr, precision) for sub_arr in arr]
    else:
        return [round(item, precision) for item in arr]


def round_weights(state_dict, precision=5):
    # Convert to JSON-serializable format (convert tensors to lists)
    return {
        key: torch.tensor(round_list(value.tolist(), precision))
        for key, value in state_dict.items()
        # key: value.tolist() for key, value in state_dict.items()
    }
