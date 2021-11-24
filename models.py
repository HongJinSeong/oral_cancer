import torch
from torch import nn

class EfficientEnsemble_10fold(nn.Module):
    def __init__(self,
                 model,
                 model_list,
                 num_classes=2):
        super(EfficientEnsemble_10fold, self).__init__()

        self.model = model
        self.model_list = model_list

    def forward(self, values):
        self.model.load_state_dict(torch.load(self.model_list[0]))
        output = self.model(values)
        for path in self.model_list[1:]:
            self.model.load_state_dict(torch.load(path))
            output+=self.model(values)

        return output