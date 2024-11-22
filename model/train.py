import torch
import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder
from model.classifier import Classifier

from model.utils import weight_init

class Trainer(nn.Module):
    def __init__(self, model_type='small', task: str = "", num_class='int'):
        super().__init__()
        if model_type == 'tiny':
            embed_dim = 192
        elif model_type == 'small':
            embed_dim = 512
        else:
            assert False, r'Trainer: check the vit model type'

        self.encoder = Encoder(task)

        if task == 'train':
            self.decoder = Decoder(in_dim=[64, 128, 256, embed_dim])
        elif task == 'classifier':
            self.decoder = Classifier(in_dim=[64, 128, 256, embed_dim], num_class=num_class)
        weight_init(self.decoder)

    def forward(self, x, y):
        fx, fy = self.encoder(x, y)
        c2x, c3x, c4x, c5x = fx
        c2y, c3y, c4y, c5y = fy
        # print(f'Initial shapes - c5x: {c5x.shape}, c4x: {c4x.shape}, c3x: {c3x.shape}, c2x: {c2x.shape}')
        pred = self.decoder(fx, fy)

        return pred
