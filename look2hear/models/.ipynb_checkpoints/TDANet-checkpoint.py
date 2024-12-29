
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_model import BaseModel

from look2hear.datas.lrs2datamodule import LRS2DataModule
import look2hear.datas
import look2hear.models
import look2hear.system
import look2hear.losses
import look2hear.metrics
import look2hear.utils
from look2hear.utils.parser_utils import (
    prepare_parser_from_dict,
    parse_args_as_dict,
)
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob

    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """ Assumes input of size `[batch, chanel, *]`. """
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x):
        """ Applies forward pass.

        Works for any input size > 2D.

        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`

        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + 1e-8).sqrt())


class ConvNormAct(nn.Module):
    """
    This class defines the convolution layer with normalization and a PReLU
    activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=True, groups=groups
        )
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(nn.Module):
    """
    This class defines the convolution layer with normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1, bias=True):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=bias, groups=groups
        )
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class NormAct(nn.Module):
    """
    This class defines a normalization and PReLU activation
    """

    def __init__(self, nOut):
        """
        :param nOut: number of output channels
        """
        super().__init__()
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.norm(input)
        return self.act(output)


class DilatedConv(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        self.conv = nn.Conv1d(
            nIn,
            nOut,
            kSize,
            stride=stride,
            dilation=d,
            padding=((kSize - 1) // 2) * d,
            groups=groups,
        )

    def forward(self, input):
        return self.conv(input)


class DilatedConvNorm(nn.Module):
    """
    This class defines the dilated convolution with normalized output.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        self.conv = nn.Conv1d(
            nIn,
            nOut,
            kSize,
            stride=stride,
            dilation=d,
            padding=((kSize - 1) // 2) * d,
            groups=groups,
        )
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class FFN(nn.Module):
    def __init__(self, in_features, hidden_size, drop=0.1):
        super().__init__()
        self.fc1 = ConvNorm(in_features, hidden_size, 1, bias=False)
        self.dwconv = nn.Conv1d(
            hidden_size, hidden_size, 5, 1, 2, bias=True, groups=hidden_size
        )
        self.act = nn.ReLU()
        self.fc2 = ConvNorm(hidden_size, in_features, 1, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, in_channels, max_length):
        pe = torch.zeros(max_length, in_channels)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(
            (
                torch.arange(0, in_channels, 2, dtype=torch.float)
                * -(math.log(10000.0) / in_channels)
            )
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super().__init__()
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, n_head, dropout, is_casual):
        super().__init__()
        self.pos_enc = PositionalEncoding(in_channels, 10000)
        self.attn_in_norm = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(in_channels, n_head, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(in_channels)
        self.is_casual = is_casual

    def forward(self, x):
        x = x.transpose(1, 2)
        attns = None
        output = self.pos_enc(self.attn_in_norm(x))
        output, _ = self.attn(output, output, output)
        output = self.norm(output + self.dropout(output))
        return output.transpose(1, 2)

class GA(nn.Module):
    def __init__(self, in_chan, out_chan, drop_path) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(out_chan, 8, 0.1, False)
        self.mlp = FFN(out_chan, out_chan * 2, drop=0.1)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x

class LA(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int = 1) -> None:
        super().__init__()
        groups = 1
        if inp == oup:
            groups = inp
        self.local_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_act = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        """
        x_g: global features
        x_l: local features
        """
        B, N, T = x_l.shape
        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=T, mode="nearest")

        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=T, mode="nearest")

        out = local_feat * sig_act + global_feat
        return out


class UConvBlock(nn.Module):
    """
    This class defines the block which performs successive downsampling and
    upsampling in order to be able to analyze the input features in multiple
    resolutions.
    """

    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1, stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(
            DilatedConvNorm(
                in_channels, in_channels, kSize=5, stride=1, groups=in_channels, d=1
            )
        )

        for i in range(1, upsampling_depth):
            if i == 0:
                stride = 1
            else:
                stride = 2
            self.spp_dw.append(
                DilatedConvNorm(
                    in_channels,
                    in_channels,
                    kSize=2 * stride + 1,
                    stride=stride,
                    groups=in_channels,
                    d=1,
                )
            )
        
        self.loc_glo_fus = nn.ModuleList([])
        for i in range(upsampling_depth):
            self.loc_glo_fus.append(LA(in_channels, in_channels))
        
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)

        self.globalatt = GA(
            in_channels * upsampling_depth, in_channels, 0.1
        )
        self.last_layer = nn.ModuleList([])
        for i in range(self.depth - 1):
            self.last_layer.append(LA(in_channels, in_channels, 5))

    def forward(self, x):
        """
        :param x: input feature map
        :return: transformed feature map
        """
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]

        # Do the downsampling process from the previous level
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        # global features
        global_f = torch.zeros(
            output[-1].shape, requires_grad=True, device=output1.device
        )
        for fea in output:
            global_f = global_f + F.adaptive_avg_pool1d(
                fea, output_size=output[-1].shape[-1]
            )
        global_f = self.globalatt(global_f)  # [B, N, T]

        x_fused = []
        # Gather them now in reverse order
        for idx in range(self.depth):
            local = output[idx]
            x_fused.append(self.loc_glo_fus[idx](local, global_f))
        
        # print(len(x_fused))
        expanded = None
        for i in range(self.depth - 2, -1, -1):
            if i == self.depth - 2:
                expanded = self.last_layer[i](x_fused[i], x_fused[i - 1])
            else:
                expanded = self.last_layer[i](x_fused[i], expanded)


        
        # x_fused = []
        # # Gather them now in reverse order
        # for idx in range(self.depth):
        #     # print(idx)
        #     if idx in [0,1,2,3]:  # 只保留第1, 2, 4层
        #         local = output[idx]
        #         x_fused.append(self.loc_glo_fus[idx](local, global_f))

        # expanded = None
        # for i in range(len(x_fused) - 2, -1, -1):
        #     if i == len(x_fused) - 2:
        #         expanded = self.last_layer[i](x_fused[i], x_fused[i-1])
        #     else:
        #         expanded = self.last_layer[i](x_fused[i], expanded)


        return self.res_conv(expanded) + residual


class Recurrent(nn.Module):
    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4, _iter=4):
        super().__init__()
        self.unet = UConvBlock(out_channels, in_channels, upsampling_depth)
        self.iter = _iter
        self.concat_block = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, 1, groups=out_channels), nn.PReLU()
        )

    def forward(self, x):
        mixture = x.clone()
        for i in range(self.iter):
            if i == 0:
                x = self.unet(x)
            else:
                x = self.unet(self.concat_block(mixture + x))
        return x


class TDANet(BaseModel):
    def __init__(
        self,
        out_channels=128,
        in_channels=512,
        num_blocks=16,
        upsampling_depth=4,
        enc_kernel_size=21,
        num_sources=2,
        sample_rate=16000,
    ):
        super(TDANet, self).__init__(sample_rate=sample_rate)

        # Number of sources to produce
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.enc_kernel_size = enc_kernel_size * sample_rate // 1000
        self.enc_num_basis = self.enc_kernel_size // 2 + 1
        self.num_sources = num_sources

        # Appropriate padding is needed for arbitrary lengths
        self.lcm = abs(
            self.enc_kernel_size // 4 * 4 ** self.upsampling_depth
        ) // math.gcd(self.enc_kernel_size // 4, 4 ** self.upsampling_depth)

        # Front end
        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=self.enc_num_basis,
            kernel_size=self.enc_kernel_size,
            stride=self.enc_kernel_size // 4,
            padding=self.enc_kernel_size // 2,
            bias=False,
        )
        torch.nn.init.xavier_uniform_(self.encoder.weight)

        # Norm before the rest, and apply one more dense layer
        self.ln = GlobLN(self.enc_num_basis)
        self.bottleneck = nn.Conv1d(
            in_channels=self.enc_num_basis, out_channels=out_channels, kernel_size=1
        )

        # Separation module
        self.sm = Recurrent(out_channels, in_channels, upsampling_depth, num_blocks)

        mask_conv = nn.Conv1d(out_channels, num_sources * self.enc_num_basis, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

        # Back end
        self.decoder = nn.ConvTranspose1d(
            in_channels=self.enc_num_basis * num_sources,
            out_channels=num_sources,
            kernel_size=self.enc_kernel_size,
            stride=self.enc_kernel_size // 4,
            padding=self.enc_kernel_size // 2,
            groups=1,
            bias=False,
        )
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        self.mask_nl_class = nn.ReLU()

    def pad_input(self, input, window, stride):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, window - stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    # Forward pass
    def forward(self, input_wav):
        # input shape: (B, T)
        was_one_d = False
        if input_wav.ndim == 1:
            was_one_d = True
            input_wav = input_wav.unsqueeze(0)
        if input_wav.ndim == 2:
            input_wav = input_wav
        if input_wav.ndim == 3:
            input_wav = input_wav.squeeze(1)

        x, rest = self.pad_input(
            input_wav, self.enc_kernel_size, self.enc_kernel_size // 4
        )
        # Front end
        x = self.encoder(x.unsqueeze(1))

        # Split paths
        s = x.clone()
        # Separation module
        x = self.ln(x)
        x = self.bottleneck(x)
        x = self.sm(x)

        x = self.mask_net(x)
        x = x.view(x.shape[0], self.num_sources, self.enc_num_basis, -1)
        x = self.mask_nl_class(x)
        x = x * s.unsqueeze(1)
        # Back end
        estimated_waveforms = self.decoder(x.view(x.shape[0], -1, x.shape[-1]))
        estimated_waveforms = estimated_waveforms[
            :,
            :,
            self.enc_kernel_size
            - self.enc_kernel_size
            // 4 : -(rest + self.enc_kernel_size - self.enc_kernel_size // 4),
        ].contiguous()
        if was_one_d:
            return estimated_waveforms.squeeze(0)
        return estimated_waveforms

    def get_model_args(self):
        model_args = {"n_src": 2}
        return model_args
    

# 定义计算稀疏度函数
def calculate_sparsity(model):
    """
    计算模型的稀疏度。
    :param model: 模型
    :return: 稀疏率
    """
    total_params = 0
    zero_params = 0
    # for name in model:
    #     param = model[name]
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    sparsity = zero_params / total_params
    print(f"Model sparsity: {sparsity * 100:.2f}%")
    return sparsity

# 定义推理时间测量函数
def measure_inference_time(model, input_size=(1, 1, 16000), device='cuda'):
    """
    测量模型的单次推理时间。
    :param model: 模型
    :param input_size: 输入数据的形状
    :param device: 运行设备
    :return: 推理时间
    """
    model.eval()
    input_data = torch.randn(input_size).to(device)
    model = model.to(device)

    # GPU 时间测量工具
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        # 热身运行
        for _ in range(10):
            _ = model(input_data)
        
        # 测量推理时间
        starter.record()
        _ = model(input_data)
        ender.record()
        torch.cuda.synchronize()
        inference_time = starter.elapsed_time(ender)  # 毫秒
    print(f"Inference time: {inference_time:.2f} ms")
    return inference_time

# 定义精度评估函数
def evaluate_accuracy(model, dataloader, criterion, device='cuda'):
    """
    在测试集上评估模型的精度。
    :param model: 模型
    :param dataloader: 测试数据加载器
    :param criterion: 损失函数
    :param device: 运行设备
    :return: 平均损失
    """
    model.eval()
    model = model.to(device)
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    
    average_loss = total_loss / total_samples
    print(f"Test loss: {average_loss:.4f}")
    return average_loss

def prune(W, X, s):
    """
    自定义剪枝方法，修复索引越界问题。
    :param W: 权重矩阵 (C_out, C_in, kernel_size)
    :param X: 输入矩阵 (N, C_in, L)
    :param s: 剪枝比例 (0 < s < 1)
    :return: 剪枝后的权重矩阵
    """
    # 确保输入和权重在 CPU 上
    W = W.cpu()
    X = X.cpu()

    # 计算 X 在时间维度上的范数，形状变为 (C_in,)
    X_norm = X.pow(2).sum(dim=(0, 2)).sqrt()

    # 计算剪枝指标 (C_out, C_in, kernel_size)
    metric = W.abs() * X_norm.unsqueeze(-1)

    # 展平 kernel_size 维度
    metric = metric.view(W.size(0), -1)

    # 根据剪枝指标排序
    _, sorted_idx = torch.sort(metric, dim=1)

    # 确定要剪枝的数量
    num_prune = int(metric.size(1) * s)
    pruned_idx = sorted_idx[:, :num_prune]

    # 创建掩码并剪枝
    mask = torch.ones_like(W.view(W.size(0), -1))

    # 遍历每个输出通道，更新对应的掩码
    for i in range(pruned_idx.size(0)):  # 遍历每个输出通道
        valid_indices = pruned_idx[i]  # 当前通道的索引
        valid_indices = valid_indices[valid_indices < mask.size(1)]  # 确保索引在范围内
        mask[i, valid_indices] = 0  # 设置对应位置为 0

    # 恢复掩码形状并应用
    mask = mask.view_as(W)
    W.mul_(mask)

    return W


def apply_custom_pruning(model, input_data, s=0.5):
    """
    对模型中所有Conv1d层应用自定义剪枝方法。
    :param model: 目标模型
    :param input_data: 示例输入数据 (用于计算X)
    :param s: 剪枝比例 (0 < s < 1)
    """
    model.eval()  # 设置为评估模式
    hooks = []

    def forward_hook(module, input, output):
        """
        钩子函数，用于获取中间层的输入。
        """
        if isinstance(module, nn.Conv1d):
            with torch.no_grad():
                W = module.weight  # 卷积核权重
                X = input[0]  # 输入特征图
                W_pruned = prune(W, X, s)  # 调用修复后的剪枝函数
                module.weight.copy_(W_pruned)

    # 注册前向钩子
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            hooks.append(module.register_forward_hook(forward_hook))

    # 运行一次前向传播，触发钩子
    with torch.no_grad():
        _ = model(input_data)

    # 移除钩子
    for hook in hooks:
        hook.remove()

import time
def measure_inference_time_cpu(model, input_size=(1, 1, 16000), num_runs=10):
    """
    测量模型在 CPU 上的平均推理时间。
    :param model: PyTorch 模型
    :param input_size: 输入数据的形状
    :param num_runs: 测试的运行次数（取平均）
    :return: 平均推理时间（毫秒）
    """
    # 准备模型和输入数据
    model.eval()
    model = model.to("cpu")  # 确保模型在 CPU 上
    input_data = torch.randn(input_size).to("cpu")  # 创建随机输入数据

    # 热身运行，避免初始化影响
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_data)

    # 测量多次推理的耗时
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()  # 开始计时
            _ = model(input_data)
            end_time = time.time()  # 结束计时
            times.append((end_time - start_time) * 1000)  # 转换为毫秒

    avg_time = sum(times) / len(times)  # 计算平均耗时
    print(f"Average inference time on CPU: {avg_time:.2f} ms")
    return avg_time

# 测试
if __name__ == '__main__':
    device = torch.device('cuda')
    model = TDANet(
        out_channels=128,
        in_channels=512,
        num_blocks=16,
        upsampling_depth=5,
        enc_kernel_size=4,
        num_sources=2,
        sample_rate=16000,
    )
    conf = torch.load(
        '/data/stan_2024/24_2024_spearate/Look2hear/Experiments/checkpoint/TDANet/epoch=498.ckpt', map_location=device
    )  # Attempt to find the model and instantiate it.
    state_dict = conf["state_dict"]
    model.load_state_dict({k.replace('audio_model.', ''): v for k, v in state_dict.items()})
    # 模拟输入数据
    index = 0
    
    datamodule = LRS2DataModule(train_dir= 'DataPreProcess/LRS2/tr',
    valid_dir= 'DataPreProcess/LRS2/cv',
    test_dir= 'DataPreProcess/LRS2/tt',
    n_src= 2,
    sample_rate= 16000,
    segment= 2.0,
    normalize_audio= False,
    batch_size= 8,
    num_workers= 8,
    pin_memory= True,
    persistent_workers= False)
    datamodule.setup()

    train_loader, val_loader, test_loader = datamodule.make_loader
    
    # 剪枝前性能评估
    print("Before pruning:")
    calculate_sparsity(model)
    input_data = torch.randn((1,1,16000), device=device)
    measure_inference_time(model)
    

    # 应用自定义剪枝方法
    print("Applying custom pruning...")
    # 创建示例输入数据
    input_data = torch.randn((1,1,16000), device=device)
    apply_custom_pruning(model, input_data, s=0.5)

    print("After pruning:")
    # 评估剪枝后模型的性能
    measure_inference_time(model)
    calculate_sparsity(model)
    
    torch.save(model, "/data/stan_2024/24_2024_spearate/Look2hear/Experiments/checkpoint/TDANet/prune_TDANet.ckpt")
    print("Pruned model saved as pruned_afrcnn_model.pth")