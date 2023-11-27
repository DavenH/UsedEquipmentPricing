import torch

from base_model import BaseModel


class AtvModel(BaseModel):
    def __init__(self, config_json_path: str = None):
        super().__init__([
            'offset', 'displacement', 'mileage', 'sold_by_dealer',
            'has_winch', 'repairs_needed', 'age', 'age_exp', 'is_2wd'], config_json_path)

    def forward(self, x):
        current_year = 2021
        (year, cc, mileage, dealer, winch, two_wd, repairs) = x[:, 1:8].T

        prediction = (
            (
                self.offset +
                self.displacement * cc +
                self.sold_by_dealer * dealer +
                self.is_2wd * two_wd +
                self.has_winch * winch +
                self.repairs_needed * repairs
            ) *
            (self.age * torch.exp((year - current_year) * self.age_exp)) +
            self.mileage * mileage
        )

        return prediction

    def augment_loss(self, loss: torch.tensor):
        kappa = 0.1
        # regularize with ridge regression
        for param in self.parameters():
            loss.add_(torch.square(param).squeeze(0) * kappa)
