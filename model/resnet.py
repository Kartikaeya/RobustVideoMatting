import torch
from torch import nn
from torchvision.models.resnet import ResNet, Bottleneck

def load_matched_state_dict(model, state_dict, print_stats=True):
    """
    Only loads weights that matched in key and shape. Ignore other weights.
    """
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    for key in curr_state_dict.keys():
        num_total += 1
        if key in state_dict and curr_state_dict[key].shape == state_dict[key].shape:
            curr_state_dict[key] = state_dict[key]
            num_matched += 1
    model.load_state_dict(curr_state_dict)
    if print_stats:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')

class ResNet50Encoder(ResNet):
    def __init__(self, pretrained: bool = False):
        super().__init__(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            replace_stride_with_dilation=[False, False, True],
            norm_layer=None)
        
        self.conv1 = nn.Conv2d(6, 64, 7, 2, 3, bias=False)

        if pretrained:
            pretrained_state_dict = torch.hub.load_state_dict_from_url(
                'https://download.pytorch.org/models/resnet50-0676ba61.pth')
            
            # self.load_state_dict(torch.hub.load_state_dict_from_url(
            #     'https://download.pytorch.org/models/resnet50-0676ba61.pth'))
            
            load_matched_state_dict(self, pretrained_state_dict)

        del self.avgpool
        del self.fc
        
    def forward_single_frame(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f1 = x  # 1/2
        x = self.maxpool(x)
        x = self.layer1(x)
        f2 = x  # 1/4
        x = self.layer2(x)
        f3 = x  # 1/8
        x = self.layer3(x)
        x = self.layer4(x)
        f4 = x  # 1/16
        return [f1, f2, f3, f4]
    
    def forward_time_series(self, x):
        B, T = x.shape[:2]
        features = self.forward_single_frame(x.flatten(0, 1))
        features = [f.unflatten(0, (B, T)) for f in features]
        return features
    
    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)
