import torch

from base_model import BaseModel


class TrailerModel(BaseModel):
    def __init__(self, config_json_path: str = None):
        super().__init__([
            'offset',
            'length', 'length2',
            'width', 'width2',
            'capacity', 'capacity2',
            'dealer', 'no_reg', 'winch',
            'age', 'age_exp', 'rust_resistance'], config_json_path)
    def forward(self, x):
        current_year = 2021
        (length, width, capacity, age, dealer, sidewalls, no_reg, winch, rust_resistance) = x[:, 1:].T

        prediction = (
            self.offset +
            self.length * length + self.length2 * torch.square(length) +
            self.width * width + self.width2 * torch.square(width) +
            self.capacity * capacity + self.capacity2 * torch.square(capacity / 1000) +
            self.dealer * dealer +
            self.winch * winch
        ) * self.age * torch.exp((age - current_year) * self.age_exp) + self.no_reg * no_reg + self.rust_resistance * rust_resistance

        return prediction

    def augment_loss(self, loss: torch.tensor):
        kappa = 0.05
        # regularize with ridge regression
        for param in self.parameters():
            loss.add_(torch.square(param).squeeze(0) * kappa)

