import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def update_ema_variables(model, ema_model, alpha, global_step=1):
    # alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)  # add_(other， alpha)为torch.add()的in-place版， 直接替换，加上other * alpha

def show_param(model):
    parameters = model.state_dict()
    print(parameters.values())

def change_param(model, value):
    parameters = model.state_dict()
    parameters['fc.weight'] = torch.ones((3, 4)).float() * value
    parameters['fc.bias'] = torch.zeros((3, )).float()
    model.load_state_dict(parameters)
    return model

def show_alpha_change(alpha_constant, epochs):
    alpha_list = []
    for global_step in range(epochs):
        alpha = min(1 - 1 / (global_step + 1), alpha_constant)
        alpha_list.append(alpha)
    plt.plot(range(epochs), alpha_list)
    plt.show()


class MODEL(nn.Module):
    def __init__(self):
        super(MODEL, self).__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc.forward(x)


if __name__ == "__main__":
    see_updata = False
    see_alpha_change = True

    if see_updata:
        model = MODEL()
        ema_model = MODEL()
        alpha = 0.5

        change_param(model, 4)
        change_param(ema_model, 2)

        update_ema_variables(model, ema_model, alpha)

        show_param(ema_model)
    
    if see_alpha_change:
        show_alpha_change(0.999, 800)