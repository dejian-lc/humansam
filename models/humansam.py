import torch
import torch.nn as nn


class HumanSAM(nn.Module):
    def __init__(self, model1, model2, num_classes, num_sample=1):
        super(HumanSAM, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.num_sample = num_sample

        for param in self.model2.parameters():
            param.requires_grad = False

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(2816, 1024)

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.head = nn.Linear(1024, num_classes)

    def forward(self, x1, x2):

        features1 = self.model1(x1)

        flag = False

        if self.num_sample > 1 and x2.size(0) % self.num_sample == 0:
            split_size = x2.size(0) // self.num_sample
            x2 = x2[:split_size]
            flag = True

        outputs = []
        # print(x2.shape)
        for t in range(x2.shape[2]):
 
            if t == 0 or t == 3:
                current_feature = x2[:, :, t, :, :]  # (bs, 3, 1536, 1536)
                output = self.model2(current_feature)  # (bs, 1024, 48, 48)
                outputs.append(output)

        outputs = torch.stack(outputs, dim=2)  # (bs, 1024, 2, 48, 48)

        averaged_output = torch.mean(outputs, dim=2)  # (bs, 1024, 48, 48)

        pooled_feature = self.global_avg_pool(averaged_output)  # (bs, 1024, 1, 1)
        features2 = pooled_feature.view(pooled_feature.size(0), -1)  # (bs, 1024)

        if flag:
            features2 = features2.repeat(self.num_sample, 1)


        features1 = self.fc1(features1) 
        hfr_features = self.alpha * features1 + (1 - self.alpha) * features2  # (bs, 1024)

        # Classification
        output = self.head(hfr_features)
        return output

    def get_num_layers(self):
        return len(self.model1.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed',
            'pos_embed_spatial',
            'pos_embed_temporal',
            'pos_embed_cls',
            'cls_token'
        }
