import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

from base_model import BaseModel


class ExcavatorModel(BaseModel):
    def __init__(self, device: torch.device, makes_series=None, config_json_path: str = None):
        super().__init__(['offset', 'is_canadian', 'has_metal_tracks', 'has_shoes',
                           'has_thumb', 'n_buckets', 'has_closed_cab', 'sale_year',
                           'engine_hp', 'engine_hp_exp', 'mass', 'mass_exp',
                           'condition', 'hours_exp', 'is_sold', 'age_exp', 'scale_factor'], config_json_path)
        le = LabelEncoder()
        encoded_makes = le.fit_transform(makes_series)
        n_classes = len(le.classes_)
        self.one_hot_makes = nn.functional.one_hot(torch.as_tensor(encoded_makes), num_classes=n_classes).float().to(device)

        if 'makes' in self.starting_values:
            self.makes = nn.Parameter(torch.tensor(self.starting_values['makes'], dtype=torch.float32))
        else:
            self.makes = nn.Parameter(torch.randn(n_classes))

    def forward(self, x):
        (arg_year, arg_sale_year, arg_hours, arg_sold, arg_cad,
         arg_metal, arg_shoes, arg_thumb, arg_buckets,
         arg_closed, arg_cond, arg_hp, arg_mass) = x[:, 2:15].T

        age = arg_sale_year - arg_year
        current_year = 2022.5
        prediction = (
            (
                self.offset * 1000 +
                self.is_canadian * arg_cad +
                self.has_metal_tracks * arg_metal +
                self.has_shoes * arg_shoes +
                self.has_thumb * arg_thumb +
                self.n_buckets * arg_buckets +
                self.has_closed_cab * arg_closed +
                self.sale_year * (current_year - arg_sale_year) +
                self.engine_hp * torch.pow(arg_hp / 10, self.engine_hp_exp) +
                self.mass * torch.pow(arg_mass / 1000, self.mass_exp)
            ) *
            (
                10000 * self.scale_factor * torch.exp(
                    age * self.age_exp / 5 +
                    arg_hours * self.hours_exp / 100 +
                    self.condition * arg_cond / 100
                )
            ) *
            (1 + self.is_sold * arg_sold) *
            (1 + self.one_hot_makes.mm(self.makes.t().unsqueeze(1)).squeeze(1))
        )
        return prediction

    def augment_loss(self, loss: torch.tensor):
        # a penalty for drifting out of a zero-mean for the makes - this way they can be compared more easily.
        loss.add_(torch.square(self.makes.sum() * 1000) * 1)

        # regularize the magnitude of these, so the 'make' factor doesn't
        # cancel out any other factors if there's low data counts for this brand
        loss.add_(torch.square(self.makes).sum() * 100000)

    def augment_saved_params(self, trained_params: dict):
        trained_params['makes'] = self.makes.tolist()