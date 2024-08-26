import torch
import torch.nn as nn
import torch.nn.functional as F

def init_branch_weights(module, method='xavier'):
    if isinstance(module, nn.Linear):
        if method == 'xavier':
            nn.init.xavier_uniform_(module.weight)
        elif method == 'kaiming':
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1)
                                            for in_channels in in_channels_list])
        self.fpn_convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                                        for _ in in_channels_list])

    def forward(self, inputs):
        # Generate lateral features
        lateral_features = [lateral_conv(x) for x, lateral_conv in zip(inputs, self.lateral_convs)]
        
        # Fuse features with upsampling
        fpn_features = []
        for i in range(len(lateral_features) - 1, -1, -1):
            if i == len(lateral_features) - 1:
                fpn_features.append(self.fpn_convs[i](lateral_features[i]))
            else:
                # Upsample the previous feature map to match the current lateral feature map size
                upsampled_feature = F.interpolate(fpn_features[-1], size=lateral_features[i].shape[2:], mode='nearest')
                
                # Ensure the dimensions match exactly
                if upsampled_feature.shape[2:] != lateral_features[i].shape[2:]:
                    diff_y = lateral_features[i].shape[2] - upsampled_feature.shape[2]
                    diff_x = lateral_features[i].shape[3] - upsampled_feature.shape[3]
                    
                    # Pad or crop the upsampled feature map to match dimensions
                    upsampled_feature = F.pad(upsampled_feature, (diff_x // 2, diff_x - diff_x // 2,
                                                                diff_y // 2, diff_y - diff_y // 2))
                    
                fpn_features.append(self.fpn_convs[i](lateral_features[i] + upsampled_feature))
        
        # Return fused features in the original order
        return fpn_features[::-1]


class DynamicFeatureFusion(nn.Module):
    def __init__(self, in_channels, num_branches):
        super(DynamicFeatureFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels * num_branches, in_channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, num_branches, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, branch_outputs):
        # Align the spatial dimensions of the feature maps
        min_height = min([feat.size(2) for feat in branch_outputs])
        min_width = min([feat.size(3) for feat in branch_outputs])
        
        resized_outputs = [F.interpolate(feat, size=(min_height, min_width), mode='nearest') for feat in branch_outputs]

        # Concatenate along the channel dimension
        stacked_outputs = torch.cat(resized_outputs, dim=1)
        
        # Attention and weighted sum
        weights = self.attention(stacked_outputs)
        weighted_sum = sum(w * out for w, out in zip(weights.split(1, dim=1), resized_outputs))
        return weighted_sum



# Self attention
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# Fusing the different branches. Like ensembling models, but doing it online
class FusionNet(nn.Module):
    def __init__(self, num_classes=10):
        super(FusionNet, self).__init__()
        # Convolutional layers (Bottom-up pathway)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Residual blocks
        self.residual_block1 = ResidualBlock(32, 64)
        self.residual_block2 = ResidualBlock(64, 128)
        self.residual_block3 = ResidualBlock(128, 256)
        self.residual_block4 = ResidualBlock(256, 512)

        # Self attention blocks -> used after residual blocks 3 and 4
        self.se_block1 = SEBlock(256)
        self.se_block2 = SEBlock(512)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        self.spatial_dropout = nn.Dropout2d(p=0.2)


        # TODO - I am bit still playing with this
        # Feature maps concatenated for each branch
        # Branch 1 will use the output from residual_block4 (high-level features)
        fm_concats_branch1 = 256 + 512 + 256 # Output channels of residual_block4

        # Branch 2 will use the output from conv1 and residual_block1 (low-level features)
        fm_concats_branch2 = 32 + 64  # Output channels of conv1 and residual_block1

        fm_concats_branch3 = 256 # Output of channels of conv2 after droput as c6
        
        fm_concats_branch4 = 256 # Output of fusion out channel

        # Branch 1: Focuses on high-level features - (from the deepest layers of the network)
        self.branch1_fc1 = nn.Linear(fm_concats_branch1, 256)
        self.branch1_fc2 = nn.Linear(256, num_classes)
        self.branch1_fc1.apply(lambda m: init_branch_weights(m, 'xavier'))
        self.branch1_fc2.apply(lambda m: init_branch_weights(m, 'xavier'))

        # Branch 2: Focuses on low-level features - (from the earlier layers of the network)
        # Attention mechanism for branch 2
        self.attention_branch2 = nn.Sequential(
            nn.Linear(fm_concats_branch2, fm_concats_branch2 // 8),
            nn.ReLU(),
            nn.Linear(fm_concats_branch2 // 8, fm_concats_branch2),
            nn.Sigmoid()
        )
        self.branch2_fc1 = nn.Linear(fm_concats_branch2, 128)
        self.branch2_fc2 = nn.Linear(128, num_classes)
        self.branch2_fc1.apply(lambda m: init_branch_weights(m, 'kaiming'))
        self.branch2_fc2.apply(lambda m: init_branch_weights(m, 'kaiming'))

        

        # Higher-level features (from the last convolutional layer after residual blocks)
        self.branch3_fc = nn.Linear(fm_concats_branch3, num_classes)

        # FPN (Feature Pyramid Network)
        self.fpn = FPN(in_channels_list=[64, 128, 256, 512], out_channels=256)

        # Dynamic Feature Fusion - Specify the number of branches (from FPN, it's 4 branches)
        self.dynamic_fusion = DynamicFeatureFusion(in_channels=256, num_branches=4)

        # Fully connected layers after fusion
        self.branch4_fc1 = nn.Linear(fm_concats_branch4, 128)
        self.branch4_fc2 = nn.Linear(128, num_classes)
        self.branch4_fc1.apply(lambda m: init_branch_weights(m, 'xavier'))
        self.branch4_fc2.apply(lambda m: init_branch_weights(m, 'xavier'))


        

    def forward(self, x, epoch=None, train=False):
        # Bottom-up pathway
        # Initial convo and pooling
        c1 = self.conv1(x)              # Input (3x32x32), Output (16x32x32)
        c1p = self.pool(c1)             # Output (16x16x16)
        
        # Residual blocks
        c2 = self.residual_block1(c1p)  # Output (64x16x16)
        c2p = self.pool(c2)             # Output (64x8x8)
        
        c3 = self.residual_block2(c2p)  # Output (128x8x8)
        c3p = self.pool(c3)             # Output (128x4x4)
        
        c4 = self.residual_block3(c3p)  # Output (256x4x4)
        c4 = self.se_block1(c4)         # Self attention
        
        c5 = self.residual_block4(c4)   # Output (512x4x4)
        c5 = self.se_block2(c5)         # Self attention

        
        # last convo layer
        c6 = self.conv2(c5)             # Input (128x4x4), Output (256x4x4)

        # spatial dropout
        c6 = self.spatial_dropout(c6)

        

        # Global average pooling for each branch
        # TODO making the GAP out of c5 only yields in better results probably
        #gap_branch1 = F.adaptive_avg_pool2d(c5, 1).squeeze(-1).squeeze(-1)  # High-level features
        gap_branch1 = torch.cat([
            F.adaptive_avg_pool2d(c4, 1).squeeze(-1).squeeze(-1),  # Low-level features
            F.adaptive_avg_pool2d(c5, 1).squeeze(-1).squeeze(-1),
            F.adaptive_avg_pool2d(c6, 1).squeeze(-1).squeeze(-1)
        ], dim=1)
        
        # Branch 1: Process high-level features
        branch1_out = F.relu(self.branch1_fc1(gap_branch1))
        branch1_out = self.branch1_fc2(branch1_out)
        
        
        gap_branch2 = torch.cat([
            F.adaptive_avg_pool2d(c1p, 1).squeeze(-1).squeeze(-1),  # Low-level features
            F.adaptive_avg_pool2d(c2p, 1).squeeze(-1).squeeze(-1)
        ], dim=1)

        # Branch 2: Process low-level features
        attention_weights_branch2 = self.attention_branch2(gap_branch2)
        branch2_out = gap_branch2 * attention_weights_branch2
        branch2_out = F.relu(self.branch2_fc1(branch2_out))
        branch2_out = self.branch2_fc2(branch2_out)

        
        # Branch 3: process higher level feature
        gap_branch3 = F.adaptive_avg_pool2d(c6, 1).squeeze(-1).squeeze(-1)  # High-level features
        branch3_out = self.branch3_fc(gap_branch3)

        # Branch 4: FPN
        # Apply FPN
        fpn_features = self.fpn([c2p, c3p, c4, c5])
        # Dynamic feature fusion
        fusion_out = self.dynamic_fusion(fpn_features)
        # GAP and FC for branch 4
        gap_branch4 = F.adaptive_avg_pool2d(fusion_out, 1).squeeze(-1).squeeze(-1)
        branch4_out = self.dropout(F.relu(self.branch4_fc1(gap_branch4)))
        branch4_out = self.branch4_fc2(branch4_out)

        # weights: # TODO this can be maybe learned or expereminted with
        branch1_weight = 5 # GAP of High Level featured (c4, c5, c6) --> FC1, FC2
        branch2_weight = 2 # GAP of Low Level features (c1, c2) --> Self attention -> FC1, FC2
        branch3_weight = 3 # High level feature from c6 --> FC
        branch4_weight = 5 # FPN and Fusion out of c2, c3, c4, c5 ->FC1, FC2

        final_out = (branch1_out*branch1_weight + branch2_out*branch2_weight + branch3_out*branch3_weight + branch4_out*branch4_weight) / (branch1_weight + branch2_weight + branch3_weight+ branch4_weight)
        return final_out
