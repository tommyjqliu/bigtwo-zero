import torch
import numpy as np

from train.models import BigtwoModel
from env.env import Observation


def _load_model(model_path):
    model = BigtwoModel("cuda:0")
    model_state_dict = model.state_dict()
    if torch.cuda.is_available():
        pretrained = torch.load(model_path, map_location="cuda:0")
    else:
        pretrained = torch.load(model_path, map_location="cpu")

    pretrained = {k: v for k, v in pretrained.items() if k in model_state_dict}
    model_state_dict.update(pretrained)
    model.load_state_dict(model_state_dict)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


class DeepAgent:

    def __init__(self, model_path):
        self.model = _load_model(model_path)

    def act(self, _obs: Observation):
        if len(_obs.legal_actions) == 1:
            return _obs.legal_actions[0]

        obs = _obs.to_tensor("cuda:0")
        y_pred = self.model.forward(obs.x_batch, return_value=True)[
            "values"
        ]
        y_pred = y_pred.detach().cpu().numpy()
        best_action_index = np.argmax(y_pred, axis=0)[0]
        print(f"Predicted values:", _obs.legal_actions[best_action_index].code)
        return _obs.legal_actions[best_action_index]


class EvaluateAgent:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def act(self, _obs: Observation):
        if len(_obs.legal_actions) == 1:
            return _obs.legal_actions[0]

        obs = _obs.to_tensor("cuda:0")
        with torch.no_grad():
            y_pred = self.model.forward(obs.z_batch, obs.x_batch, return_value=True)[
                "values"
            ]
        y_pred = y_pred.detach().cpu().numpy()
        best_action_index = np.argmax(y_pred, axis=0)[0]
        return _obs.legal_actions[best_action_index]
