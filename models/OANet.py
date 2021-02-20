from models.common import * 
import torch
import torch.nn as nn
import torch.nn.functional as F


class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
                nn.InstanceNorm1d(in_channel, eps=1e-3),
                nn.BatchNorm1d(in_channel),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channel, output_points, kernel_size=1))
        
    def forward(self, x):
        embed = self.conv(x) # [bs, num_cluster, num_corr]
        S = torch.softmax(embed, dim=2)
        out = torch.matmul(x, S.transpose(1,2))
        return out

class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
                nn.InstanceNorm1d(in_channel, eps=1e-3),
                nn.BatchNorm1d(in_channel),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channel, output_points, kernel_size=1))
        
    def forward(self, x_up, x_down):
        #x_up: [bs, num_channels, num_corr]
        #x_down: [bs, num_channels, num_cluster]
        embed = self.conv(x_up)# [bs, num_cluster, num_corr]
        S = torch.softmax(embed, dim=1)
        out = torch.matmul(x_down, S)
        return out # [bs, num_channels, num_corr]

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
                nn.InstanceNorm1d(channels, eps=1e-3),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(channels, out_channels, kernel_size=1),#b*c*n*1
                Transpose(1,2))
        # Spatial Correlation Layer
        self.conv2 = nn.Sequential(
                nn.BatchNorm1d(points),
                nn.ReLU(inplace=True),
                nn.Conv1d(points, points, kernel_size=1)
                )
        self.conv3 = nn.Sequential(        
                Transpose(1,2),
                nn.InstanceNorm1d(out_channels, eps=1e-3),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_channels, out_channels, kernel_size=1)
                )
    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out


class OANet(nn.Module):
    def __init__(self, 
                in_dim=6,
                num_layers=6,
                num_channels=128, 
                num_clusters=10,
                act_pos='post',
                ):
        super(OANet, self).__init__()
        assert act_pos == 'pre' or act_pos == 'post'

        self.num_channels = num_channels
        self.num_layers = num_channels
        self.sigma = nn.Parameter(torch.Tensor([1.0]).float(), requires_grad=True)

        modules = [nn.Conv1d(in_dim, num_channels, kernel_size=1, bias=True)] 
        for i in range(num_layers//2):
            if act_pos == 'pre':
                modules.append(ContextNormalization())
                modules.append(nn.BatchNorm1d(num_channels))
                modules.append(nn.ReLU(inplace=True))
                # modules.append(EdgeConv(num_channels, num_channels, k=10))
                modules.append(nn.Conv1d(num_channels, num_channels, kernel_size=1, bias=True))
            else:
                modules.append(nn.Conv1d(num_channels, num_channels, kernel_size=1, bias=True))
                # modules.append(EdgeConv(num_channels, num_channels, k=10))
                modules.append(ContextNormalization())
                modules.append(nn.BatchNorm1d(num_channels))
                modules.append(nn.ReLU(inplace=True))
        self.l1_1 = nn.Sequential(*modules)

        modules = []
        for i in range(num_layers//2):
            modules.append(OAFilter(num_channels, num_clusters))
        self.l2 = nn.Sequential(*modules)

        self.down1 = diff_pool(num_channels, num_clusters)
        self.up1 = diff_unpool(num_channels, num_clusters)

        modules = [nn.Conv1d(num_channels*2, num_channels, kernel_size=1, bias=True)] 
        for i in range(num_layers//2-1):
            if act_pos == 'pre':
                modules.append(ContextNormalization())
                modules.append(nn.BatchNorm1d(num_channels))
                modules.append(nn.ReLU(inplace=True))
                # modules.append(EdgeConv(num_channels, num_channels, k=10))
                modules.append(nn.Conv1d(num_channels, num_channels, kernel_size=1, bias=True))
            else:
                modules.append(nn.Conv1d(num_channels, num_channels, kernel_size=1, bias=True))
                # modules.append(EdgeConv(num_channels, num_channels, k=10))
                modules.append(ContextNormalization())
                modules.append(nn.BatchNorm1d(num_channels))
                modules.append(nn.ReLU(inplace=True))
        self.l1_2 = nn.Sequential(*modules)

        self.output = nn.Conv1d(num_channels, 1, kernel_size=1)

    def forward(self, data):
        corr_pos = data['corr_pos'].permute(0, 2, 1)
        # corr_pos [bs, num_corr, in_dim]
        # corr_pos = corr_pos.permute(0,2,1)
        x1_1 = self.l1_1(corr_pos)
        x_down = self.down1(x1_1)
        x2 = self.l2(x_down)
        x_up = self.up1(x1_1, x2)

        out = self.l1_2(torch.cat([x1_1, x_up], dim=1))
        return out

        logits = self.output(out).squeeze(1)

        if len(torch.where(logits>0)[1]) >= 3:
            R, t = rigid_transform_3d(
                A=src_keypts[:, torch.where(logits>0)[1], :], 
                B=tgt_keypts[:, torch.where(logits>0)[1], :], 
                scores=torch.relu(torch.tanh( logits[:,torch.where(logits>0)[1]] )),
            )
        else:
            R = torch.eye(3)[None,:,:].to(corr_pos.device)
            t = torch.ones(1, 3)[None,:, :].to(corr_pos.device)
        
        R = torch.cat([R, torch.zeros_like(R[:, 0:1, :])], dim=1)
        t = t.permute(0,2,1)
        t = torch.cat([t, torch.ones_like(t[:, 0:1, :])], dim=1)
        trans = torch.cat([R, t], dim=-1)

        res = {
            "final_trans": trans, 
            "final_labels": logits, 
            "M": None
        }
        return res
