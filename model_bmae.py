from functools import partial
import torch
import torch.nn as nn
import copy
from timm.models.vision_transformer import Block, PatchEmbed

from util.pos_embed import get_2d_sincos_pos_embed
from models_mae import MaskedAutoencoderViT


class BootstrapMAE(MaskedAutoencoderViT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Projection layer adaptation for feature prediction
        self.feat_projection = nn.Linear(kwargs['decoder_embed_dim'], kwargs['embed_dim'])
        nn.init.xavier_normal_(self.feat_projection.weight)
        if self.feat_projection.bias is not None:
            nn.init.zeros_(self.feat_projection.bias)

        self.teacher = {}  # Teacher model parameters
        self.ema_enabled = kwargs.get('enable_ema', False)
        if self.ema_enabled:
            self.ema_init_epochs = kwargs['ema_warmup_epochs']
            self._init_teacher_weights()
            self.ema_momentum = kwargs['ema_alpha']
            self.current_epoch = 0

    def _init_teacher_weights(self):
        """Initialize teacher model with current weights"""
        components = ['patch_embed', 'cls_token', 'pos_embed', 'blocks', 'norm']
        for comp in components:
            self.teacher[comp] = copy.deepcopy(getattr(self, comp)).cuda()

    def _update_teacher_component(self, teacher_comp, model_comp):
        """EMA update for individual components"""
        if isinstance(teacher_comp, nn.Module):
            for t_param, m_param in zip(teacher_comp.parameters(), model_comp.parameters()):
                t_param.data.mul_(self.ema_momentum).add_(m_param.data, alpha=1 - self.ema_momentum)
        else:  # Handle tensors (pos_embed, cls_token)
            teacher_comp.data = teacher_comp.data * self.ema_momentum + model_comp.data * (1 - self.ema_momentum)

    def update_teacher(self):
        """Update teacher model weights using EMA"""
        components = ['patch_embed', 'cls_token', 'pos_embed', 'blocks', 'norm']
        for comp in components:
            self._update_teacher_component(self.teacher[comp], getattr(self, comp))

    def feature_decoder_forward(self, latent, restore_ids):
        """Decoder forward pass for feature prediction"""
        x = self.decoder_embed(latent)

        # Generate mask tokens and reconstruct full sequence
        num_masked = restore_ids.shape[1] + 1 - x.shape[1]
        mask_tokens = self.mask_token.expand(x.size(0), num_masked, -1)
        x_body = torch.cat([x[:, 1:], mask_tokens], dim=1)

        # Restore original patch order
        x_body = torch.gather(x_body, 1, restore_ids.unsqueeze(-1).expand(-1, -1, x.size(2)))
        x = torch.cat([x[:, :1], x_body], dim=1)

        # Process through decoder
        x += self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return x[:, 1:]  # Remove CLS token

    def get_teacher_features(self, images):
        """Get encoded features from teacher model"""
        patches = self.teacher['patch_embed'](images)
        patches += self.teacher['pos_embed'][:, 1:]

        # Add CLS token
        cls_token = self.teacher['cls_token'] + self.teacher['pos_embed'][:, :1]
        cls_tokens = cls_token.expand(patches.size(0), -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)

        # Process through transformer
        for blk in self.teacher['blocks']:
            x = blk(x)
        return self.teacher['norm'](x)

    def feature_prediction_loss(self, teacher_feats, pred_feats, mask):
        """Calculate feature reconstruction loss"""
        if self.norm_pix_loss:
            teacher_feats = (teacher_feats - teacher_feats.mean(-1, keepdim=True)) / (
                    teacher_feats.var(-1, keepdim=True, unbiased=False) + 1e-6).sqrt()
        return (mask * (pred_feats - teacher_feats).pow(2).mean(-1)).sum() / mask.sum()

    def track_epoch(self):
        """Update epoch counter for EMA scheduling"""
        self.current_epoch += 1

    def forward(self, inputs, mask_ratio=0.75):
        """Main forward pass with mode switching"""
        if self.ema_enabled:
            self.update_teacher()

        # Determine training phase
        teacher_active = (self.ema_enabled and self.current_epoch >= self.ema_init_epochs) or (
                not self.ema_enabled and self.teacher)

        if teacher_active:
            # Feature prediction mode
            latent, mask, restore_ids = self.forward_encoder(inputs, mask_ratio)
            pred_features = self.feature_decoder_forward(latent, restore_ids)
            with torch.no_grad():
                target_features = self.get_teacher_features(inputs)
            loss = self.feature_prediction_loss(target_features[:, 1:], pred_features, mask)
            return loss, pred_features, mask
        else:
            # Pixel reconstruction mode
            return super().forward(inputs, mask_ratio)

    def sync_teacher(self):
        """Full weight copy for non-EMA mode"""
        self._init_teacher_weights()


def deit_tiny(**kwargs):
    return BootstrapMAE(
        img_size=32, patch_size=4, embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=192, decoder_depth=8, decoder_num_heads=3,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)