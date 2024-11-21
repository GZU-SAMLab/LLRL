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

class TrainerClassifierFeatureVision(nn.Module):
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
            self.decoder = ClassifierFeatureVision(in_dim=[64, 128, 256, embed_dim], num_class=num_class)
        weight_init(self.decoder)

    def forward(self, x, y):
        fx, fy = self.encoder(x, y)
        c2x, c3x, c4x, c5x = fx
        c2y, c3y, c4y, c5y = fy
        # print(f'Initial shapes - c5x: {c5x.shape}, c4x: {c4x.shape}, c3x: {c3x.shape}, c2x: {c2x.shape}')
        pred = self.decoder(fx, fy)

        return pred


class TrainerPNA(nn.Module):
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
            self.decoder = ClassifierNA(in_dim=[64, 128, 256, embed_dim], num_class=num_class)
        weight_init(self.decoder)

    def forward(self, x, y):
        fx, fy = self.encoder(x, y)
        c2x, c3x, c4x, c5x = fx
        c2y, c3y, c4y, c5y = fy
        # print(f'Initial shapes - c5x: {c5x.shape}, c4x: {c4x.shape}, c3x: {c3x.shape}, c2x: {c2x.shape}')
        pred = self.decoder(fx, fy)

        return pred

class TrainerNIA(nn.Module):
    def __init__(self, model_type='small', task: str = "", num_class='int'):
        super().__init__()
        if model_type == 'tiny':
            embed_dim = 192
        elif model_type == 'small':
            embed_dim = 512
        else:
            assert False, r'Trainer: check the vit model type'

        self.encoder = EncoderNIA(task)

        if task == 'train':
            self.decoder = DecoderNIA(in_dim=[64, 128, 256, embed_dim])
        elif task == 'classifier':
            self.decoder = ClassifierNIA(in_dim=[64, 128, 256, embed_dim], num_class=num_class)
        weight_init(self.decoder)

    def forward(self, x, y):
        fx, fy = self.encoder(x, y)
        # print(f'Initial shapes - c5x: {c5x.shape}, c4x: {c4x.shape}, c3x: {c3x.shape}, c2x: {c2x.shape}')
        pred = self.decoder(fx, fy)

        return pred

class TrainerNHFE(nn.Module):
    def __init__(self, model_type='small', task: str = "", num_class='int'):
        super().__init__()
        if model_type == 'tiny':
            embed_dim = 192
        elif model_type == 'small':
            embed_dim = 512
        else:
            assert False, r'Trainer: check the vit model type'

        self.encoder = EncoderNHFE(task)

        if task == 'train':
            self.decoder = DecoderNHFE(in_dim=[64, 128, 256, embed_dim])
        elif task == 'classifier':
            self.decoder = ClassifierNHFE(in_dim=[64, 128, 256, embed_dim], num_class=num_class)
        weight_init(self.decoder)

    def forward(self, x, y):
        fx, fy = self.encoder(x, y)
        # print(f'Initial shapes - c5x: {c5x.shape}, c4x: {c4x.shape}, c3x: {c3x.shape}, c2x: {c2x.shape}')
        pred = self.decoder(fx, fy)

        return pred

class TrainerNCFFA(nn.Module):
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
            self.decoder = DecoderNCFFA(in_dim=[64, 128, 256, embed_dim])
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

class TrainerClassifierNCFFA(nn.Module):
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
            self.decoder = DecoderNCFFA(in_dim=[64, 128, 256, embed_dim])
        elif task == 'classifier':
            self.decoder = ClassifierNCFFA(in_dim=[64, 128, 256, embed_dim], num_class=num_class)
        weight_init(self.decoder)

    def forward(self, x, y):
        fx, fy = self.encoder(x, y)
        c2x, c3x, c4x, c5x = fx
        c2y, c3y, c4y, c5y = fy
        # print(f'Initial shapes - c5x: {c5x.shape}, c4x: {c4x.shape}, c3x: {c3x.shape}, c2x: {c2x.shape}')
        pred = self.decoder(fx, fy)

        return pred

class TrainerNoPreIA(nn.Module):
    def __init__(self, model_type='small', task: str = "", num_class='int'):
        super().__init__()
        if model_type == 'tiny':
            embed_dim = 192
        elif model_type == 'small':
            embed_dim = 512
        else:
            assert False, r'Trainer: check the vit model type'

        self.encoder = EncoderNoPreIA(task)

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

class FeatureVision(nn.Module):
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
            self.decoder = FeatureVisionModel1(in_dim=[64, 128, 256, embed_dim])
        elif task == 'classifier':
            self.decoder = FeatureVisionModel(in_dim=[64, 128, 256, embed_dim], num_class=num_class)
        weight_init(self.decoder)

    def forward(self, x, y):
        fx, fy = self.encoder(x, y)
        c2x, c3x, c4x, c5x = fx
        c2y, c3y, c4y, c5y = fy
        # print(f'Initial shapes - c5x: {c5x.shape}, c4x: {c4x.shape}, c3x: {c3x.shape}, c2x: {c2x.shape}')
        fused_feature = self.decoder(fx, fy)

        return fused_feature
