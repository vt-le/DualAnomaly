import torch
import torch.nn as nn
from models.unet import UNet
from models.pvt2 import pvt_v2_b2, pvt_v2_b2_flow
# from models.unet_dec import UNetDec
from models.DSCG_module import DSCG


class AnomalyDetection(nn.Module):
    def __init__(self, config, logger, is_trained=True):
        super().__init__()
        self.concatenated_layers = config.MODEL.CONCATENATED_LAYERS
        num_frames = config.MODEL.NUM_FRAMES
        self.reconstruction = UNet(config=config)

        self.encoder_of = pvt_v2_b2_flow()
        self.encoder_ofh = pvt_v2_b2_flow()
        self.encoder_fr = pvt_v2_b2()
        if (num_frames == 4):
            number_frames = [4, 4, 4, 4, 4, 4]
        else:
            number_frames = [5, 5, 5, 5, 5, 5]
        if len(self.concatenated_layers) == 4:
            self.decoder = DSCG(number_frames=number_frames,
                                num_convs=2,
                                num_heads=[2, 4, 4, 8, 16],
                                mlp_ratios=[4, 4, 4, 4, 4],
                                # norm_cfg = dict(type='GN', num_groups=2),
                                # act_cfg=dict(type='LeakyReLU')
                                # sr_ratios=[1, 1, 1, 1, 1],
                                depths=[1, 1, 1, 1, 1]
                                )
        elif len(self.concatenated_layers) == 1:
            self.decoder = DSCG(number_frames=number_frames)

        # Initialize
        if is_trained:
            self.encoder_of.init_weights(logger, config.MODEL.PRETRAINED)
            self.encoder_ofh.init_weights(logger, config.MODEL.PRETRAINED)
            self.encoder_fr.init_weights(logger, config.MODEL.PRETRAINED)

    def forward(self, fr, of):
        # reconstruction
        out_of = self.reconstruction(of)

        # Prediction
        condition_h = self.encoder_ofh(out_of)
        condition = self.encoder_of(of)
        features = []
        for f in fr:
            features.append(self.encoder_fr(f))

        # features.append(condition)

        # Transpose list of lists => https://stackoverflow.com/questions/6473679/transpose-list-of-lists
        features = list(map(list, zip(*features)))

        for i in range(len(features)):
            feature_i = features[i]
            if i not in self.concatenated_layers:
                feature_i = feature_i[:-1]

            features[i] = torch.cat([fea for fea in feature_i], dim=1)

        out_fr = self.decoder(features, condition, condition_h)

        return out_fr, out_of

    def get_attention_map(self, im):
        condition = self.encoder_of(im)
        return condition


