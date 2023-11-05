import torch
from torch import Tensor
from torch import nn
from torchsummary import summary
from torch.nn import functional as F
from typing import Optional, List

from .mobilenetv3 import MobileNetV3LargeEncoder
from .resnet import ResNet50Encoder
from .lraspp import LRASPP
from .decoder import RecurrentDecoder, Projection
from .fast_guided_filter import FastGuidedFilterRefiner
from .deep_guided_filter import DeepGuidedFilterRefiner
from .refiner import Refiner
class MattingNetwork(nn.Module):
    def __init__(self,
                 variant: str = 'mobilenetv3',
                 refiner: str = 'patch_based',
                 pretrained_backbone: bool = False):
        super().__init__()
        assert variant in ['mobilenetv3', 'resnet50']
        assert refiner in ['fast_guided_filter', 'deep_guided_filter', 'patch_based']
        
        if variant == 'mobilenetv3':
            self.backbone = MobileNetV3LargeEncoder(pretrained_backbone)
            self.aspp = LRASPP(960, 128)
            self.decoder = RecurrentDecoder([16, 24, 40, 128], [80, 40, 32, 16])
        else:
            self.backbone = ResNet50Encoder(pretrained_backbone)
            self.aspp = LRASPP(2048, 256)
            self.decoder = RecurrentDecoder([64, 256, 512, 256], [128, 64, 32, 16])
            
        self.project_mat = Projection(16, 5)
        self.project_seg = Projection(16, 1)
        self.refiner_name = refiner
        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        elif refiner == 'patch_based':
            self.refiner = Refiner(mode = "sampling", sample_pixels=80_000, threshold= 0.7, kernel_size=3)
        else:
            self.refiner = FastGuidedFilterRefiner()
        
    def forward(self,
                src: Tensor,
                r1: Optional[Tensor] = None,
                r2: Optional[Tensor] = None,
                r3: Optional[Tensor] = None,
                r4: Optional[Tensor] = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):
        
        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
        else:
            src_sm = src
        
        f1, f2, f3, f4 = self.backbone(src_sm)
        f4 = self.aspp(f4)
        hid, *rec = self.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4)
        
        if not segmentation_pass:
            fgr_residual_sm, pha_sm, err_sm = self.project_mat(hid).split([3, 1, 1], dim=-3)

            pha_sm = pha_sm.clamp(0., 1.)
            # fgr_residual_sm = fgr_residual_sm
            err_sm = err_sm.clamp(0.,1.)
            hid_sm = hid.relu()

            fgr_residual_lg = None
            fgr_lg = None
            pha_lg = None

            # refine matting pass here
            if downsample_ratio != 1:

                assert src.size(3) // 4 * 4 == src.size(3) and src.size(4) // 4 * 4 == src.size(4), \
                    'src and bgr must have width and height that are divisible by 4'
                
                if self.refiner_name == "patch_based":
                    B, T = src.shape[:2]

                    fgr_residual_lg, pha_lg, _ = self.refiner(src.flatten(0, 1), pha_sm.flatten(0,1), fgr_residual_sm.flatten(0, 1), err_sm.flatten(0, 1), hid_sm.flatten(0, 1))
                    
                    fgr_residual_lg = fgr_residual_lg.unflatten(0, (B, T))
                    pha_lg = pha_lg.unflatten(0, (B, T))
                else:
                    fgr_residual_lg, pha_lg = self.refiner(src[:, :, :3, ...], src_sm[:, :, :3, ...], fgr_residual_sm, pha_sm, hid)
            
            fgr_sm = fgr_residual_sm + src_sm[:, :, :3, ...]
            fgr_sm = fgr_sm.clamp(0., 1.)

            if fgr_residual_lg != None:
                fgr_lg = fgr_residual_lg + src[:, :, :3, ...]
                fgr_lg = fgr_lg.clamp(0., 1.)
                pha_lg = pha_lg.clamp(0., 1.)

            return [pha_sm, fgr_sm, err_sm, pha_lg, fgr_lg, *rec]
        else:
            seg = self.project_seg(hid)
            return [seg, *rec]

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x
