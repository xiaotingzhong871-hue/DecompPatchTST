# models/DecompPatchTST.py
import torch
import torch.nn as nn
import math


class SeriesDecomposition(nn.Module):

    def __init__(self, kernel_size):
        super(SeriesDecomposition, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) 
        )

    def forward(self, x):
        trend = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B,S,V]
        seasonal = x - trend
        return seasonal, trend


class PolynomialTrend(nn.Module):

    def __init__(self, seq_len, pred_len, n_vars, degree=2):
        super(PolynomialTrend, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.degree = degree

        t_in = torch.linspace(-1, 1, steps=seq_len)  
        t_out = torch.linspace(-1, 1, steps=pred_len)  

        X_in = torch.stack([t_in ** i for i in range(degree + 1)], dim=1)  # [S, D+1]
        X_out = torch.stack([t_out ** i for i in range(degree + 1)], dim=1)  # [L, D+1]

        self.register_buffer("X_in", X_in)  # [S, D+1]
        self.register_buffer("X_out", X_out)  # [L, D+1]

        self.coeff_transform = nn.Linear(degree + 1, degree + 1)

    def forward(self, x_trend):

        B, S, V = x_trend.shape
        assert V == self.n_vars, f"输入变量数 {V} 与初始化 {self.n_vars} 不一致"

        x_trend = x_trend.permute(0, 2, 1).reshape(B * V, S)

        pseudo_inv = torch.linalg.pinv(self.X_in)  # [D+1, S]
        coeff = torch.matmul(pseudo_inv, x_trend.T).T  # [B*V, D+1]

        coeff = self.coeff_transform(coeff)  # [B*V, D+1]

        trend_out = torch.matmul(self.X_out, coeff.T).T  # [B*V, L]

        trend_out = trend_out.reshape(B, V, self.pred_len).permute(0, 2, 1)

        return trend_out


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.use_norm = configs.use_norm

        self.decomposition = SeriesDecomposition(configs.moving_avg)

        self.patch_len = 16
        self.stride = 8
        self.num_patches = (self.seq_len - self.patch_len) // self.stride + 1

        self.patch_embedding = nn.Linear(self.patch_len, self.d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.d_model,
            nhead=8,
            dim_feedforward=configs.d_ff,
            dropout=configs.dropout,
            activation=configs.activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=configs.e_layers
        )

        self.head_dropout = nn.Dropout(configs.dropout)
        self.projection_seasonal = nn.Linear(self.d_model, self.pred_len)

        self.projection_trend = PolynomialTrend(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            n_vars=self.enc_in,
            degree=3 
        )

        self.projection_noise = nn.Linear(self.d_model, self.pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x_enc /= stdev

        seasonal_init, trend_init = self.decomposition(x_enc)

        trend_output = self.projection_trend(trend_init)  # [B, L, V]

        seasonal_init = seasonal_init.permute(0, 2, 1)  # [B, V, S]
        seasonal_patched = seasonal_init.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        B, V, Np, Pl = seasonal_patched.shape
        seasonal_patched = seasonal_patched.reshape(-1, Np, Pl)

        embedded_patches = self.patch_embedding(seasonal_patched)
        embedded_patches += self.pos_embedding

        encoded_output = self.transformer_encoder(embedded_patches)
        prediction_features = encoded_output[:, -1, :]
        prediction_features = self.head_dropout(prediction_features)

        seasonal_output_flat = self.projection_seasonal(prediction_features)
        seasonal_output = seasonal_output_flat.reshape(B, V, self.pred_len)
        seasonal_output = seasonal_output.permute(0, 2, 1)  # [B, L, V]

        noise_output_flat = self.projection_noise(prediction_features)
        noise_output = noise_output_flat.reshape(B, V, self.pred_len)
        noise_output = noise_output.permute(0, 2, 1)  # [B, L, V]

        dec_out = trend_output + seasonal_output + noise_output

        if self.use_norm:
            dec_out = dec_out * stdev.repeat(1, self.pred_len, 1)
            dec_out = dec_out + means.repeat(1, self.pred_len, 1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]


