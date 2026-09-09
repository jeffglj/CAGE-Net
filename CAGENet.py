"""
CAGE-Net: Two-View Correspondence Pruning via Cascaded Adaptive
          Geometry-Enhanced Learning
================================================================
Built on the SPD (OANet family) backbone. Four coordinated designs:

  CGCA-RS - Conflict-Gated Coherence Attention with Reliability Stabilization
          Upgrades PGDA by treating local consensus as a hypothesis that must
          be verified by global epipolar compatibility. The post-GEGR path uses
          the learned w0 confidence as a weak certification seed to stabilize
          in-module E estimation on pseudo-consensus-heavy indoor scenes.

  GEGR  - Geometry-Enhanced Graph Reasoning
          GCN Block (node-axis aggregation) + CPT module
          (channel-axis coordinate-conditioned attention + MBFFN).

  LGFE  - Local-Global Feature Enhancement
          ResNet/GNN/OA backbone with MSDA densely inserted.
          
  MSDA  - Multi-Scale Difference Attention (4 branches).

  CSMF  - Cross-Stage Multi-scale Feature Fusion
          Attention-weighted fusion of stage-0 key-node features,
          injected into stage 1.

Main network class: CAGENet  (alias FusedCLNet kept for backward
compatibility with existing training scripts).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import batch_episym


# ============================================================
# Basic utilities
# ============================================================

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class ResNetBlock(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNetBlock, self).__init__()
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
    device = X.device
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.linalg.eigh(X[batch_idx, :, :].squeeze(), UPLO='U')
        bv[batch_idx, :, :] = v
    bv = bv.to(device)
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


# ============================================================
# KNN / Graph helpers
# ============================================================

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


# ============================================================
# SPD backbone blocks
# ============================================================

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
            Transpose(1, 2))
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(points),
            nn.ReLU(),
            nn.Conv2d(points, points, kernel_size=1)
        )
        self.conv3 = nn.Sequential(
            Transpose(1, 2),
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


class DiffPool(nn.Module):
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


class DiffUnpool(nn.Module):
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
        self.down1 = DiffPool(channels, l2_nums)
        self.l2 = []
        for _ in range(self.layer_num // 2):
            self.l2.append(OAFilter(channels, l2_nums))
        self.up1 = DiffUnpool(channels, l2_nums)
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


class GCNBlock(nn.Module):
    def __init__(self, in_channel):
        super(GCNBlock, self).__init__()
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


# ============================================================
# Module 1: CGCA-RS - Original CGCA with Reliability Stabilization
#   Restores the original CGCA carrier that gave the strongest YFCC curve,
#   and adds only a minimal SUN3D-oriented stabilizer:
#     1) non-zero bounded conflict gain, so the conflict branch can learn;
#     2) post-GEGR seed certification for the in-module E weights.
# ============================================================

class ConflictGatedAttention(nn.Module):
    """Original CGCA attention with a bounded reliability-stabilized E seed."""
    def __init__(self, dim, heads=4, dim_head=32,
                 attn_dropout=0.0, proj_dropout=0.0,
                 fuse_gain_init=0.01, fuse_gain_max=0.05,
                 seed_mix=0.35):
        super().__init__()
        assert heads * dim_head == dim, "heads * dim_head must equal dim"
        self.heads = heads
        self.dim_head = dim_head
        self.attn_dropout = attn_dropout
        self.fuse_gain_max = float(fuse_gain_max)
        self.seed_mix = float(seed_mix)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_dropout)

        self.geo_dim = 16
        self.geo_enc = nn.Sequential(
            nn.Conv2d(2, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.geo_dim, 1),
        )
        self.geo_alpha = nn.Parameter(torch.zeros(1))

        hidden = max(dim // 4, 8)
        self.local_head = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.global_head = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.fuse_proj = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.fuse_proj[-1].weight)
        nn.init.zeros_(self.fuse_proj[-1].bias)
        self.fuse_gain = nn.Parameter(torch.tensor(float(fuse_gain_init)))

    def _zscore(self, value):
        mean = value.mean(dim=1, keepdim=True)
        std = value.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        return ((value - mean) / std).clamp(-4.0, 4.0)

    def _local_coherence(self, q, k, motion):
        B, H, N, Dh = q.shape
        qh = q.mean(1)
        kh = k.mean(1)
        sim = torch.einsum("bnd,bmd->bnm", qh, kh) / (Dh ** 0.5)
        attn = torch.softmax(sim, dim=-1)
        mean = torch.bmm(attn, motion)
        second = torch.bmm(attn, motion * motion)
        var = (second - mean * mean).sum(-1).clamp_min(0.0)
        var = var / (var.mean(dim=1, keepdim=True) + 1e-6)
        return var

    def _seed_weight(self, local_prob, seed_score):
        w = local_prob.squeeze(-1).detach()
        if seed_score is None:
            return self._zscore(w)
        seed = seed_score.detach()
        if seed.dim() == 3:
            seed = seed.squeeze(-1)
        seed = torch.sigmoid(seed.reshape_as(w))
        mixed = (1.0 - self.seed_mix) * w + self.seed_mix * seed
        return self._zscore(mixed)

    def _estimate_E_and_epi(self, coords4, weight_logit):
        B, N, _ = coords4.shape
        x1 = coords4[..., 0]
        y1 = coords4[..., 1]
        x2 = coords4[..., 2]
        y2 = coords4[..., 3]
        ones = torch.ones_like(x1)
        A = torch.stack([
            x2 * x1, x2 * y1, x2,
            y2 * x1, y2 * y1, y2,
            x1, y1, ones,
        ], dim=-1)
        wn = torch.softmax(weight_logit, dim=1).unsqueeze(-1)
        M = torch.bmm(A.transpose(1, 2), A * wn)
        M = M + 1e-4 * torch.eye(9, device=M.device).unsqueeze(0)
        try:
            _, evecs = torch.linalg.eigh(M)
            e = evecs[:, :, 0]
        except RuntimeError:
            return torch.zeros(B, N, device=coords4.device)

        E = e.view(B, 3, 3)
        p1 = torch.stack([x1, y1, ones], dim=-1)
        p2 = torch.stack([x2, y2, ones], dim=-1)
        Ep1 = torch.bmm(p1, E.transpose(1, 2))
        Etp2 = torch.bmm(p2, E)
        x2Ex1 = (p2 * Ep1).sum(-1)
        den = (Ep1[..., 0] ** 2 + Ep1[..., 1] ** 2
               + Etp2[..., 0] ** 2 + Etp2[..., 1] ** 2 + 1e-9)
        epi = (x2Ex1 ** 2) / den
        epi = epi / (epi.mean(dim=1, keepdim=True) + 1e-6)
        return epi

    def forward(self, x, coords=None, seed_score=None, use_global=True,
                return_aux=False):
        B, N, C = x.shape
        H, Dh = self.heads, self.dim_head
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, H, Dh).transpose(1, 2)
        k = k.view(B, N, H, Dh).transpose(1, 2)
        v = v.view(B, N, H, Dh).transpose(1, 2)

        bias = None
        if coords is not None:
            disp = coords[:, 2:4, :, :] - coords[:, 0:2, :, :]
            geo = self.geo_enc(disp).squeeze(-1)
            geo = F.normalize(geo, dim=1, eps=1e-6)
            bias = (self.geo_alpha * torch.einsum("bdn,bdm->bnm", geo, geo)).unsqueeze(1)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        if coords is not None:
            coords4 = coords[:, 0:4, :, :].squeeze(-1).transpose(1, 2).contiguous()
            motion = coords4[..., 2:4] - coords4[..., :2]
            var = self._local_coherence(q, k, motion)
            local_prob = torch.sigmoid(-self.local_head(var.unsqueeze(-1)))
            if use_global:
                weight_logit = self._seed_weight(local_prob, seed_score)
                epi = self._estimate_E_and_epi(coords4, weight_logit)
                global_prob = torch.sigmoid(-self.global_head(epi.unsqueeze(-1)))
            else:
                global_prob = local_prob.detach()
        else:
            var = None
            local_prob = x.new_zeros(B, N, 1)
            global_prob = local_prob

        lg = torch.cat([local_prob, global_prob], dim=-1)
        gain = self.fuse_gain.clamp(0.0, self.fuse_gain_max)
        out = out + gain * self.fuse_proj(lg)

        if return_aux:
            return out, var, local_prob.squeeze(-1), global_prob.squeeze(-1)
        return out


class CGCABlock(nn.Module):
    def __init__(self, dim=128, heads=4, dim_head=32, mlp_ratio=2.0,
                 attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ConflictGatedAttention(
            dim, heads, dim_head, attn_dropout, proj_dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(proj_dropout),
        )

    def forward(self, x, coords=None, seed_score=None, use_global=True,
                return_aux=False):
        if return_aux:
            attn, coh, local_prob, global_prob = self.attn(
                self.norm1(x), coords=coords, seed_score=seed_score,
                use_global=use_global, return_aux=True)
            x = x + attn
            x = x + self.ffn(self.norm2(x))
            return x, coh, local_prob, global_prob
        x = x + self.attn(
            self.norm1(x), coords=coords, seed_score=seed_score,
            use_global=use_global)
        x = x + self.ffn(self.norm2(x))
        return x


class PGDA(nn.Module):
    """CGCA-RS drop-in replacement for PGDA. Input/output: (B, C, N, 1)."""
    def __init__(self, dim=128, depth=2, heads=4, dim_head=32, mlp_ratio=2.0,
                 attn_dropout=0.0, proj_dropout=0.0, gate_mode='input',
                 topk_ratio=0.5, k=9, state_chunk=128, min_seed_k=32,
                 state_gain_max=0.05):
        super().__init__()
        self.blocks = nn.ModuleList([
            CGCABlock(dim, heads, dim_head, mlp_ratio, attn_dropout, proj_dropout)
            for _ in range(depth)
        ])
        self.coords = None
        self.seed_score = None
        self.use_global_evidence = True
        self.last_coherence = None
        self.last_l = None
        self.last_g = None

    def forward(self, x, return_gate=False):
        x_seq = x.squeeze(-1).transpose(1, 2).contiguous()
        coh = local_prob = global_prob = None
        gates = []
        for i, blk in enumerate(self.blocks):
            if i == len(self.blocks) - 1 or return_gate:
                x_seq, coh, local_prob, global_prob = blk(
                    x_seq, coords=self.coords, seed_score=self.seed_score,
                    use_global=self.use_global_evidence, return_aux=True)
                if return_gate:
                    gates.append((local_prob * global_prob).mean(dim=1).detach())
            else:
                x_seq = blk(
                    x_seq, coords=self.coords, seed_score=self.seed_score,
                    use_global=self.use_global_evidence)
        self.last_coherence = coh
        self.last_l = local_prob
        self.last_g = global_prob
        out = x_seq.transpose(1, 2).contiguous().unsqueeze(-1)
        if return_gate:
            return out, gates
        return out


# ============================================================
# Module 2: MSDA - Multi-Scale Difference Attention
# ============================================================

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


# ============================================================
# Module 3 (part): CPT - Coordinate-conditioned channel attention + MBFFN
# ============================================================

class CPTAttention(nn.Module):
    def __init__(self, in_channels, out_channels, use_rel=True):
        super(CPTAttention, self).__init__()
        self.use_rel = use_rel

        self.coord_encoder = nn.Conv2d(2, in_channels, kernel_size=1)
        if self.use_rel:
            self.rel_encoder = nn.Conv2d(2, in_channels, kernel_size=1)
        else:
            self.rel_encoder = None

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

    def forward(self, feature, coords):
        q = self.q(feature).squeeze(3)
        k = self.k(feature).squeeze(3)
        v = self.v(feature).squeeze(3)

        coord_1 = coords[:, :2, :, :]
        coord_2 = coords[:, 2:4, :, :]
        graph_1 = self.coord_encoder(coord_1)
        graph_2 = self.coord_encoder(coord_2)
        graph_context = graph_1 + graph_2
        if self.use_rel:
            graph_context = graph_context + self.rel_encoder(coord_1 - coord_2)
        graph_context = graph_context.squeeze(3)

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


class CPT(nn.Module):
    def __init__(self, channels, use_rel=True):
        super(CPT, self).__init__()
        self.attn = CPTAttention(channels, channels, use_rel=use_rel)
        self.LayerNorm1 = nn.LayerNorm(channels, eps=1e-6)
        self.mbffn = MBFFN(channels, channels)
        self.LayerNorm2 = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, feature, coords):
        attn_feature = self.attn(feature, coords)
        attn_feature = attn_feature + feature
        ln1 = attn_feature.squeeze(3).transpose(-1, -2)
        ln1 = self.LayerNorm1(ln1)
        ln1 = ln1.transpose(-1, -2).unsqueeze(3)
        
        mbffn_feature = self.mbffn(ln1)
        
        ln2 = mbffn_feature.squeeze(3).transpose(-1, -2)
        ln2 = self.LayerNorm2(ln2)
        ln2 = ln2.transpose(-1, -2).unsqueeze(3)
        out = ln1 + ln2
        return out


# ============================================================
# LGFE sub-modules: MLPs / AFF / GNN
# ============================================================

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
        out = get_graph_feature(features, k=self.knn_num)
        out_an = self.conv(out)
        out_an = self.change1(out_an)
        out_max = self.mlp1(out)
        out_max = out_max.max(dim=-1, keepdim=False)[0]
        out_max = out_max.unsqueeze(3)
        out_max = self.change2(out_max)
        out = self.aff(out_max, out_an) + features
        return out


# ============================================================
# Module 3: GEGR - Geometry-Enhanced Graph Reasoning
# ============================================================

class GEGR(nn.Module):
    def __init__(self, channels, use_rel=True):
        super(GEGR, self).__init__()
        self.gcn = GCNBlock(channels)
        self.cpt = CPT(channels, use_rel=use_rel)

    def forward(self, feature, weight, coords):
        out_g = self.gcn(feature, weight)
        out_gcn = out_g + feature
        out = self.cpt(out_gcn, coords)
        return out, out_gcn


# ============================================================
# Module 4: CSMF - Cross-Stage Multi-scale Feature Fusion
# ============================================================

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


# ============================================================
# CAGEStage: one pruning stage integrating LGFE + PGDA + GEGR (+ CSMF input)
# ============================================================

class CAGEStage(nn.Module):
    def __init__(self, initial=False, predict=False, out_channel=128, k_num=8,
                 sampling_rate=0.5, pgda_depth=2, cross_stage=False,
                 use_pgda=True, use_msda=True, use_gegr=True,
                 use_rel=True, gate_mode='input', pgda_state_chunk=128):
        super(CAGEStage, self).__init__()
        self.initial = initial
        self.in_channel = 4 if self.initial is True else 6
        self.out_channel = out_channel
        self.k_num = k_num
        self.predict = predict
        self.sr = sampling_rate
        self.cross_stage = cross_stage
        self.use_pgda = use_pgda
        self.use_gegr = use_gegr
        self.use_pre_global_evidence = True
        self.use_post_global_evidence = True

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )

        self.lgfe_pre = nn.Sequential(
            ResNetBlock(self.out_channel, self.out_channel, pre=False),
            ResNetBlock(self.out_channel, self.out_channel, pre=False),
            GNN(int(self.k_num), self.out_channel),
            MSDA(in_channels=self.out_channel) if use_msda else nn.Identity(),
            ResNetBlock(self.out_channel, self.out_channel, pre=False),
            ResNetBlock(self.out_channel, self.out_channel, pre=False),
            MSDA(in_channels=self.out_channel) if use_msda else nn.Identity(),
            OABlock(self.out_channel, clusters=256),
            MSDA(in_channels=self.out_channel) if use_msda else nn.Identity(),
            ResNetBlock(self.out_channel, self.out_channel, pre=False),
            ResNetBlock(self.out_channel, self.out_channel, pre=False),
        )

        if use_pgda:
            self.pgda_pre = PGDA(
                dim=self.out_channel, depth=pgda_depth, heads=4, dim_head=32,
                mlp_ratio=2.0, attn_dropout=0.0, proj_dropout=0.0,
                gate_mode=gate_mode,
                state_chunk=pgda_state_chunk,
            )
            self.pgda_post = PGDA(
                dim=self.out_channel, depth=pgda_depth, heads=4, dim_head=32,
                mlp_ratio=2.0, attn_dropout=0.0, proj_dropout=0.0,
                gate_mode=gate_mode,
                state_chunk=pgda_state_chunk,
            )
        else:
            self.pgda_pre = None
            self.pgda_post = None

        self.linear_0 = nn.Conv2d(self.out_channel, 1, (1, 1))

        if use_gegr:
            self.gegr = GEGR(self.out_channel, use_rel=use_rel)
        else:
            self.gegr = None

        self.lgfe_post = nn.Sequential(
            ResNetBlock(self.out_channel, self.out_channel, pre=False),
            ResNetBlock(self.out_channel, self.out_channel, pre=False),
            OABlock(self.out_channel, clusters=128),
            MSDA(in_channels=self.out_channel) if use_msda else nn.Identity(),
            ResNetBlock(self.out_channel, self.out_channel, pre=False),
            ResNetBlock(self.out_channel, self.out_channel, pre=False),
            MSDA(in_channels=self.out_channel) if use_msda else nn.Identity(),
            GNN(int(self.k_num), self.out_channel),
            MSDA(in_channels=self.out_channel) if use_msda else nn.Identity(),
            ResNetBlock(self.out_channel, self.out_channel, pre=False),
            ResNetBlock(self.out_channel, self.out_channel, pre=False),
        )

        self.linear_1 = nn.Conv2d(self.out_channel, 1, (1, 1))

        if self.cross_stage:
            self.cross_key_fuse = nn.Sequential(
                nn.Conv2d(self.out_channel * 2, self.out_channel, (1, 1)),
                nn.BatchNorm2d(self.out_channel),
                nn.ReLU(inplace=True),
            )

        if self.predict == True:
            self.embed_2 = ResNetBlock(self.out_channel, self.out_channel, pre=False)
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

    def forward(self, x, y, cross_key_outs=None, return_gate=False):
        B, _, N, _ = x.size()
        x_raw = x.transpose(1, 3).contiguous() 

        out = self.conv(x_raw) 

        if cross_key_outs is not None and self.cross_stage:
            out = out + self.cross_key_fuse(torch.cat([out, cross_key_outs], dim=1))

        out1 = out 

        out = self.lgfe_pre(out)
        gates = {}

        if self.pgda_pre is not None:
            self.pgda_pre.coords = x_raw
            self.pgda_pre.seed_score = None
            self.pgda_pre.use_global_evidence = self.use_pre_global_evidence
            if return_gate:
                out, g_pre = self.pgda_pre(out, return_gate=True)
                gates['pre'] = g_pre
            else:
                out = self.pgda_pre(out)

        w0 = self.linear_0(out).view(B, -1)

        if self.gegr is not None:
            out, out2 = self.gegr(out, w0.detach(), x_raw)
        else:
            out2 = out

        if self.pgda_post is not None:
            self.pgda_post.coords = x_raw
            self.pgda_post.seed_score = w0.detach()
            self.pgda_post.use_global_evidence = self.use_post_global_evidence
            if return_gate:
                out, g_post = self.pgda_post(out, return_gate=True)
                gates['post'] = g_post
            else:
                out = self.pgda_post(out)

        out = self.lgfe_post(out)
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

            if return_gate:
                return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds], key_outs, gates
            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds], key_outs
        else:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N * self.sr)]
            x_ds, y_ds, w0_ds, out = self.down_sampling(x, y, w0, indices, out, self.predict)
            out = self.embed_2(out)
            w2 = self.linear_2(out)
            e_hat = weighted_8points(x_ds, w2)
            if return_gate:
                return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds, w1_ds], e_hat, gates
            return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds, w1_ds], e_hat


# ============================================================
# CAGENet
# ============================================================

class CAGENet(nn.Module):
    def __init__(self, config):
        super(CAGENet, self).__init__()

        use_pgda = getattr(config, 'use_pgda', True)
        use_msda = getattr(config, 'use_msda', True)
        use_gegr = getattr(config, 'use_gegr', True)
        use_csmf = getattr(config, 'use_csmf', True)
        use_rel = getattr(config, 'use_rel', True)
        gate_mode = getattr(config, 'gate_mode', 'input')
        
        # 默认锁定 2 层 PGDA 块（共 8 块），优先贴近原版 SOTA 感受野。
        pgda_depth = max(2, int(getattr(config, 'pgda_depth', 2)))
        pgda_state_chunk = getattr(config, 'pgda_state_chunk', 128)

        self.ds_0 = CAGEStage(
            initial=True, predict=False,
            out_channel=128,
            k_num=9,
            sampling_rate=config.sr,
            pgda_depth=pgda_depth,
            cross_stage=False,
            use_pgda=use_pgda,
            use_msda=use_msda,
            use_gegr=use_gegr,
            use_rel=use_rel,
            gate_mode=gate_mode,
            pgda_state_chunk=pgda_state_chunk,
        )

        self.ds_1 = CAGEStage(
            initial=False, predict=True,
            out_channel=128,
            k_num=6,
            sampling_rate=config.sr,
            pgda_depth=pgda_depth,
            cross_stage=use_csmf,
            use_pgda=use_pgda,
            use_msda=use_msda,
            use_gegr=use_gegr,
            use_rel=use_rel,
            gate_mode=gate_mode,
            pgda_state_chunk=pgda_state_chunk,
        )

        if use_csmf:
            self.csmf = CSMF(channels=128, num_keys=3)
        else:
            self.csmf = None

    def forward(self, x, y, return_gate=False):
        B, _, N, _ = x.shape

        if return_gate:
            x1, y1, ws0, w_ds0, key_outs_ds0, gates0 = self.ds_0(x, y, return_gate=True)
        else:
            x1, y1, ws0, w_ds0, key_outs_ds0 = self.ds_0(x, y)

        if self.csmf is not None:
            ds0_all_feat = self.csmf(key_outs_ds0)
        else:
            ds0_all_feat = None

        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1)
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1)
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1)

        if return_gate:
            x2, y2, ws1, w_ds1, e_hat, gates1 = self.ds_1(
                x_, y1, cross_key_outs=ds0_all_feat, return_gate=True)
        else:
            x2, y2, ws1, w_ds1, e_hat = self.ds_1(x_, y1, cross_key_outs=ds0_all_feat)

        with torch.no_grad():
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat)

        if return_gate:
            all_gates = {'stage0': gates0, 'stage1': gates1}
            return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat, all_gates

        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat


# ------------------------------------------------------------
# Backward-compatible alias
# ------------------------------------------------------------
FusedCLNet = CAGENet
