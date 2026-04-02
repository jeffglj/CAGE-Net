import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import batch_episym


class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )

    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)
        out = out + x1
        return torch.relu(out)


def batch_symeig(X):
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.linalg.eigh(X[batch_idx, :, :].squeeze(), UPLO='U')
        bv[batch_idx, :, :] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    mask = logits[:, 0, :, 0]
    weights = logits[:, 1, :, 0]
    mask = torch.sigmoid(mask)
    weights = torch.exp(weights) * mask
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)

    x_shp = x_in.shape
    x_in = x_in.squeeze(1)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()

    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx[:, :, :]


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx_out = knn(x, k=k)
    else:
        idx_out = idx
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_out + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),
            trans(1, 2))
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(points),
            nn.ReLU(),
            nn.Conv2d(points, points, kernel_size=1)
        )
        self.conv3 = nn.Sequential(
            trans(1, 2),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
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


class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1)
        )

    def forward(self, x):
        embed = self.conv(x)
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3)
        return out


class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x_up, x_down):
        embed = self.conv(x_up)
        S = torch.softmax(embed, dim=1).squeeze(3)
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out


class OABlock(nn.Module):
    def __init__(self, net_channels, depth=6, clusters=250):
        nn.Module.__init__(self)
        channels = net_channels
        self.layer_num = depth
        l2_nums = clusters
        self.down1 = diff_pool(channels, l2_nums)
        self.l2 = []
        for _ in range(self.layer_num // 2):
            self.l2.append(OAFilter(channels, l2_nums))
        self.up1 = diff_unpool(channels, l2_nums)
        self.l2 = nn.Sequential(*self.l2)
        self.output = nn.Conv2d(channels, 1, kernel_size=1)
        self.shot_cut = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, data):
        x1_1 = data
        x_down = self.down1(x1_1)
        x2 = self.l2(x_down)
        x_up = self.up1(x1_1, x2)
        out = torch.cat([x1_1, x_up], dim=1)
        return self.shot_cut(out)


class GCN_Block(nn.Module):

    def __init__(self, in_channel):
        super(GCN_Block, self).__init__()
        self.in_channel = in_channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )

    def attention(self, w):
        w = torch.relu(torch.tanh(w)).unsqueeze(-1)
        A = torch.bmm(w.transpose(1, 2), w)
        return A

    def graph_aggregation(self, x, w):
        B, _, N, _ = x.size()
        with torch.no_grad():
            A = self.attention(w)
            I = torch.eye(N).unsqueeze(0).to(x.device).detach()
            A = A + I
            D_out = torch.sum(A, dim=-1)
            D = (1 / D_out) ** 0.5
            D = torch.diag_embed(D)
            L = torch.bmm(D, A)
            L = torch.bmm(L, D)
        out = x.squeeze(-1).transpose(1, 2).contiguous()
        out = torch.bmm(L, out).unsqueeze(-1)
        out = out.transpose(1, 2).contiguous()
        return out

    def forward(self, x, w):
        out = self.graph_aggregation(x, w)
        out = self.conv(out)
        return out


class _SpatialMHSA(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        assert heads * dim_head == dim, "heads * dim_head must equal dim"
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(B, N, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(B, N, self.heads, self.dim_head).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class _ChannelMSA(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        assert heads * dim_head == dim, "heads * dim_head must equal dim"
        self.heads = heads
        self.dim_head = dim_head

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.heads, self.dim_head).permute(0, 2, 3, 1)
        k = k.view(B, N, self.heads, self.dim_head).permute(0, 2, 3, 1)
        v = v.view(B, N, self.heads, self.dim_head).permute(0, 2, 3, 1)

        scale = (N ** -0.5)
        attn = (q * scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.permute(0, 3, 1, 2).contiguous().view(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class PGDA_Block(nn.Module):

    def __init__(self, dim=128, heads=4, dim_head=32, mlp_ratio=2.0,
                 attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.s_mhsa = _SpatialMHSA(dim, heads, dim_head, attn_dropout, proj_dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.c_msa = _ChannelMSA(dim, heads, dim_head, attn_dropout, proj_dropout)

        self.gate = nn.Parameter(torch.tensor(0.5))

        self.norm3 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(proj_dropout),
        )

    def forward(self, x):
        g = torch.sigmoid(self.gate)
        x = x + g * self.s_mhsa(self.norm1(x)) + (1 - g) * self.c_msa(self.norm2(x))
        x = x + self.ffn(self.norm3(x))
        return x


class PGDA_Stack(nn.Module):

    def __init__(self, dim=128, depth=2, heads=4, dim_head=32, mlp_ratio=2.0,
                 attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            PGDA_Block(dim, heads, dim_head, mlp_ratio, attn_dropout, proj_dropout)
            for _ in range(depth)
        ])

    def forward(self, x):
        x_seq = x.squeeze(-1).transpose(1, 2)
        for blk in self.blocks:
            x_seq = blk(x_seq)
        return x_seq.transpose(1, 2).unsqueeze(-1)


class MSDA(nn.Module):

    def __init__(self, in_channels=128, reduction=4, use_residual=True):
        super(MSDA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        inter_channels = int(self.in_channels // reduction)

        self.conv_in = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
            nn.GELU()
        )
        self.local_att = nn.Sequential(
            nn.Conv2d(self.out_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
        )
        self.global_att_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.out_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
        )
        self.global_att_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(self.out_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
        )
        self.global_local_diff = nn.Sequential(
            nn.Conv2d(self.out_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
        )

        self.branch_weights = nn.Parameter(torch.ones(4) * 0.25)

        self.sigmoid = nn.Sigmoid()
        self.conv_out = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        self.use_residual = use_residual

    def forward(self, x):
        input_conv = self.conv_in(x)

        local_scale = self.local_att(input_conv)
        global_avg_scale = self.global_att_avg(input_conv)
        global_max_scale = self.global_att_max(input_conv)

        global_mean = torch.mean(input_conv, dim=2, keepdim=True)
        diff_feat = input_conv - global_mean
        diff_scale = self.global_local_diff(diff_feat)

        bw = torch.softmax(self.branch_weights, dim=0)
        scale_out = bw[0] * local_scale + bw[1] * global_avg_scale + \
                    bw[2] * global_max_scale + bw[3] * diff_scale
        scale_out = self.sigmoid(scale_out)

        weighted_feat = input_conv * scale_out
        output = self.conv_out(weighted_feat)
        output = output + input_conv

        if self.use_residual:
            output = output + x

        return output


class CPT(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(CPT, self).__init__()
        self.coord_encoder = nn.Conv2d(2, in_channels, kernel_size=1)
        self.rel_encoder = nn.Conv2d(2, in_channels, kernel_size=1)

        self.q = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.k = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.v = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.temperature = torch.sqrt(torch.tensor(float(in_channels)))
        self.temperature2 = torch.sqrt(torch.tensor(float(in_channels)))

    def forward(self, PointCN1, x):

        q = self.q(PointCN1).squeeze(3)
        k = self.k(PointCN1).squeeze(3)
        v = self.v(PointCN1).squeeze(3)

        coord_1 = x[:, :2, :, :]
        coord_2 = x[:, 2:4, :, :]
        graph_1 = self.coord_encoder(coord_1)
        graph_2 = self.coord_encoder(coord_2)
        graph_rel = self.rel_encoder(coord_1 - coord_2)
        graph_context = (graph_1 + graph_2 + graph_rel).squeeze(3)

        graph_context_position = torch.matmul(q / self.temperature2, graph_context.transpose(1, 2))
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))
        attn = attn + graph_context_position
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v).unsqueeze(3)
        return output


class MBFFN(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=4):
        super(MBFFN, self).__init__()
        inter_channels = int(in_channels // reduction)

        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.local_att = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.global_att_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.sigmoid = nn.Sigmoid()
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        input_conv = self.conv_in(x)

        scale1 = self.local_att(input_conv)
        scale2 = self.global_att(input_conv)
        scale3 = self.global_att_max(input_conv)

        scale_out = scale1 + scale2 + scale3
        scale_out = self.sigmoid(scale_out)
        output_conv = self.conv_out(scale_out)
        out = output_conv + input_conv
        out = out + x
        return out


class CGA_Module(nn.Module):

    def __init__(self, channels):
        super(CGA_Module, self).__init__()
        self.CPA = CPT(channels, channels)
        self.LayerNorm1 = nn.LayerNorm(channels, eps=1e-6)
        self.MBFFN = MBFFN(channels, channels)
        self.LayerNorm2 = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, feature, position_feature):

        CPT_feature = self.CPA(feature, position_feature)
        CPT_feature = CPT_feature + feature
        CPT_feature_LN1 = CPT_feature.squeeze(3).transpose(-1, -2)
        CPT_feature_LN1 = self.LayerNorm1(CPT_feature_LN1)
        CPT_feature_LN1 = CPT_feature_LN1.transpose(-1, -2).unsqueeze(3)
        MBFFN_feature = self.MBFFN(CPT_feature_LN1)
        MBFFN_feature_LN2 = MBFFN_feature.squeeze(3).transpose(-1, -2)
        MBFFN_feature_LN2 = self.LayerNorm2(MBFFN_feature_LN2)
        MBFFN_feature_LN2 = MBFFN_feature_LN2.transpose(-1, -2).unsqueeze(3)
        out = CPT_feature_LN1 + MBFFN_feature_LN2
        return out


class MLPs(nn.Module):

    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)


class AFF(nn.Module):

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()
        self.scale = nn.Parameter(torch.tensor(1.5))

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        s = torch.clamp(self.scale, 0.5, 3.0)
        xo = s * x * wei + s * residual * (1 - wei)
        return xo


class GNN(nn.Module):

    def __init__(self, knn_num=9, in_channel=128):
        super(GNN, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel
        self.mlp1 = MLPs(2 * in_channel, 2 * in_channel)
        self.change1 = MLPs(2 * in_channel, in_channel)
        self.change2 = MLPs(2 * in_channel, in_channel)
        self.aff = AFF(in_channel, 4)
        assert self.knn_num == 9 or self.knn_num == 6
        if self.knn_num == 9:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel * 2, self.in_channel * 2, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(self.in_channel * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel * 2, self.in_channel * 2, (1, 3)),
                nn.BatchNorm2d(self.in_channel * 2),
                nn.ReLU(inplace=True),
            )
        if self.knn_num == 6:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel * 2, self.in_channel * 2, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(self.in_channel * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel * 2, self.in_channel * 2, (1, 2)),
                nn.BatchNorm2d(self.in_channel * 2),
                nn.ReLU(inplace=True),
            )

    def forward(self, features):
        B, _, N, _ = features.shape
        out = get_graph_feature(features, k=self.knn_num)
        out_an = self.conv(out)
        out_an = self.change1(out_an)
        out_max = self.mlp1(out)
        out_max = out_max.max(dim=-1, keepdim=False)[0]
        out_max = out_max.unsqueeze(3)
        out_max = self.change2(out_max)
        out = self.aff(out_max, out_an) + features
        return out


class GEGR(nn.Module):

    def __init__(self, channels):
        super(GEGR, self).__init__()
        self.gcn = GCN_Block(channels)
        self.cga = CGA_Module(channels)

    def forward(self, feature, weight, position_feature):

        out_g = self.gcn(feature, weight)
        out_gcn = out_g + feature
        out = self.cga(out_gcn, position_feature)
        return out, out_gcn


class CSMF(nn.Module):

    def __init__(self, channels=128, num_keys=3):
        super(CSMF, self).__init__()
        self.substage_att = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, channels, 1),
                nn.Sigmoid(),
            ) for _ in range(num_keys)
        ])
        self.conv_adjust = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, key_outs):

        weights = [att(feat) for att, feat in zip(self.substage_att, key_outs)]
        weights_sum = sum(weights) + 1e-8
        norm_weights = [w / weights_sum for w in weights]
        fused = sum(feat * w for feat, w in zip(key_outs, norm_weights))
        return self.conv_adjust(fused)


class DS_Block(nn.Module):
    def __init__(self, initial=False, predict=False, out_channel=128, k_num=8,
                 sampling_rate=0.5, pgda_depth=2, cross_stage=False):
        super(DS_Block, self).__init__()
        self.initial = initial
        self.in_channel = 4 if self.initial is True else 6
        self.out_channel = out_channel
        self.k_num = k_num
        self.predict = predict
        self.sr = sampling_rate
        self.cross_stage = cross_stage

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )

        self.embed_0 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            GNN(int(self.k_num), self.out_channel),
            MSDA(in_channels=self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            MSDA(in_channels=self.out_channel),
            OABlock(self.out_channel, clusters=256),
            MSDA(in_channels=self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )

        self.pgda_pre = PGDA_Stack(
            dim=self.out_channel,
            depth=pgda_depth,
            heads=4,
            dim_head=32,
            mlp_ratio=2.0,
            attn_dropout=0.0,
            proj_dropout=0.0,
        )
        self.pgda_post = PGDA_Stack(
            dim=self.out_channel,
            depth=pgda_depth,
            heads=4,
            dim_head=32,
            mlp_ratio=2.0,
            attn_dropout=0.0,
            proj_dropout=0.0,
        )

        self.linear_0 = nn.Conv2d(self.out_channel, 1, (1, 1))

        self.gegr = GEGR(self.out_channel)

        self.embed_1 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            OABlock(self.out_channel, clusters=128),
            MSDA(in_channels=self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            MSDA(in_channels=self.out_channel),
            GNN(int(self.k_num), self.out_channel),
            MSDA(in_channels=self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )

        self.linear_1 = nn.Conv2d(self.out_channel, 1, (1, 1))

        if self.cross_stage:
            self.cross_key_fuse = nn.Sequential(
                nn.Conv2d(self.out_channel * 2, self.out_channel, (1, 1)),
                nn.BatchNorm2d(self.out_channel),
                nn.ReLU(inplace=True),
            )

        if self.predict == True:
            self.embed_2 = ResNet_Block(self.out_channel, self.out_channel, pre=False)
            self.linear_2 = nn.Conv2d(self.out_channel, 2, (1, 1))

    def down_sampling(self, x, y, weights, indices, features=None, predict=False):
        B, _, N, _ = x.size()
        indices = indices[:, :int(N * self.sr)]
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices)
            w_out = torch.gather(weights, dim=-1, index=indices)
        indices = indices.view(B, 1, -1, 1)

        if predict == False:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2,
                                     index=indices.repeat(1, 1, 1, 4))
            return x_out, y_out, w_out
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2,
                                     index=indices.repeat(1, 1, 1, 4))
            feature_out = torch.gather(features, dim=2,
                                       index=indices.repeat(1, 128, 1, 1))
            return x_out, y_out, w_out, feature_out

    def forward(self, x, y, cross_key_outs=None):

        B, _, N, _ = x.size()
        x_raw = x.transpose(1, 3).contiguous()

        out = self.conv(x_raw)

        if cross_key_outs is not None and self.cross_stage:
            out = out + self.cross_key_fuse(torch.cat([out, cross_key_outs], dim=1))

        out1 = out

        out = self.embed_0(out)

        out = self.pgda_pre(out)

        w0 = self.linear_0(out).view(B, -1)

        out, out2 = self.gegr(out, w0.detach(), x_raw)

        out = self.pgda_post(out)

        out = self.embed_1(out)

        w1 = self.linear_1(out).view(B, -1)

        out3 = out

        if self.predict == False:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N * self.sr)]
            x_ds, y_ds, w0_ds = self.down_sampling(x, y, w0, indices, None, self.predict)

            indices_feat = indices[:, :int(N * self.sr)].view(B, 1, -1, 1)
            key_outs = []
            for feat in [out1, out2, out3]:
                feat_ds = torch.gather(feat, dim=2,
                                       index=indices_feat.repeat(1, self.out_channel, 1, 1))
                key_outs.append(feat_ds)

            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds], key_outs
        else:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N * self.sr)]
            x_ds, y_ds, w0_ds, out = self.down_sampling(x, y, w0, indices, out, self.predict)
            out = self.embed_2(out)
            w2 = self.linear_2(out)
            e_hat = weighted_8points(x_ds, w2)
            return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds, w1_ds], e_hat


class CAGENet(nn.Module):
    def __init__(self, config):
        super(CAGENet, self).__init__()

        self.ds_0 = DS_Block(
            initial=True, predict=False,
            out_channel=128,
            k_num=9,
            sampling_rate=config.sr,
            pgda_depth=2,
            cross_stage=False,
        )

        self.ds_1 = DS_Block(
            initial=False, predict=True,
            out_channel=128,
            k_num=6,
            sampling_rate=config.sr,
            pgda_depth=2,
            cross_stage=True,
        )

        self.csmf = CSMF(channels=128, num_keys=3)

    def forward(self, x, y):
        B, _, N, _ = x.shape

        x1, y1, ws0, w_ds0, key_outs_ds0 = self.ds_0(x, y)

        ds0_all_feat = self.csmf(key_outs_ds0)

        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1)
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1)
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1)

        x2, y2, ws1, w_ds1, e_hat = self.ds_1(x_, y1, cross_key_outs=ds0_all_feat)

        with torch.no_grad():
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat)

        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat