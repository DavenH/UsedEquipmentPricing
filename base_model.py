import json
import math
import random

import torch
from torch import nn
from json import encoder

encoder.FLOAT_REPR = lambda o: format(o, '.4f')
encoder.c_make_encoder = None

class BaseModel(torch.nn.Module):
    def __init__(self, param_keys: [str], config_path: str=None):
        super().__init__()
        self.param_keys = param_keys
        self.starting_values = {}

        if config_path:
            with open(config_path, 'r') as json_file:
                self.starting_values = json.load(json_file)

        for key in self.param_keys:
            value = self.starting_values.get(key, torch.randn(1).item()) * (1 + 0.1 * random.random())
            setattr(self, key, nn.Parameter(torch.tensor(value, dtype=torch.float32)))

    def fit(self,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            data: torch.Tensor,
            labels: torch.Tensor,
            n_iterations: int) -> float:

        best_loss = 1e9

        for iteration in range(n_iterations):
            optimizer.zero_grad()
            pred = self.forward(data)
            loss = self.get_loss_fn().forward(pred, labels)
            self.augment_loss(loss)
            loss.backward()
            optimizer.step()

            if iteration % 250 == 0:
                eval_loss = torch.mean(torch.abs(pred - labels))
                best_loss = min(best_loss, eval_loss.item())

                scheduler.step(eval_loss)
                print(f"Iteration {iteration}, Loss: {math.sqrt(loss.item()):.4f}, Mean error: {eval_loss:.2f}")

                if optimizer.param_groups[0]['lr'] < 1e-5:
                    print("Terminating early because LR < 1e-6")
                    break

        print(f"Finished with best loss {best_loss:.5f}")
        return best_loss

    def get_loss_fn(self) -> torch.nn.Module:
        return nn.MSELoss()

    def save_params(self, save_dir, best_loss: float):
        trained_params = {k: getattr(self, k).item() for k in self.param_keys}
        self.augment_saved_params(trained_params)

        with open(f"{save_dir}/config_{round(best_loss)}.json", 'w') as json_file:
            json.dump(trained_params, json_file, indent=2)

    def forward(self, x):
        raise NotImplementedError

    def augment_saved_params(self, trained_params: dict):
        pass

    def augment_loss(self, loss: torch.tensor):
        pass