from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
from diffusers import SchedulerMixin
from transformers import PreTrainedModel, PreTrainedTokenizer


class BaseGuide(nn.Module, ABC):
    """Diffusion guidance wrapper base class.

    Attributes:
        embeds (dict[str, torch.Tensor]): Dictionary of text embeddings.
        t_range (Tuple[float, float]): Diffusion t-range range.
        guidance_scale (float): Classifier-free guidance weight.
        tokenizer (PreTrainedTokenizer): Diffuser's Tokenizer.
        text_encoder (PreTrainedModel): Diffuser's text encoder.
        scheduler (SchedulerMixin): Diffuser's scheduler.
        device (torch.device): Device of guidance.
        dtype (torch.dtype): Precision of guidance.
    """

    def __init__(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        t_range: Tuple[float, float],
        guidance_scale: float,
        tokenizer: PreTrainedTokenizer,
        text_encoder: PreTrainedModel,
        scheduler: SchedulerMixin,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Initialize guidance components.

        Arguments:
            t_range: Diffusion time interval.
            guidance_scale: Classifier-free guidance weight.
            tokenizer: Diffuser's Tokenizer.
            text_encoder: Diffuser's text encoder.
            scheduler: Diffuser's scheduler.
            device: Device of guidance.
            dtype: Precision of guidance.
        """
        super().__init__()

        # Set basic members
        self.device = device
        self.dtype = dtype

        self.t_range = t_range
        self.guidance_scale = guidance_scale

        # Set diffusion components
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

        # Make prompt embeddings
        self.embeds = {
            "top": self.encode_text("overhead view of " + prompt),
            "front": self.encode_text("front view of " + prompt),
            "side": self.encode_text("side view of " + prompt),
            "back": self.encode_text("back view of " + prompt),
            "neg": self.encode_text(negative_prompt),
        }

    def calculate_sds_loss(
        self, render: torch.Tensor, theta: float, phi: float
    ) -> torch.Tensor:
        """Calculate score distillation sampling loss.

        Arguments:
            render: Render tensor of shape (N, *render_shape).
            theta: Render azimuth.
            phi: Render zenith.

        Returns:
            SDS loss tensor of shape (N,).
        """
        N, _, _, _ = render.shape

        # Add noise
        timesteps = (
            (
                torch.rand((N,), device=self.device)
                * (self.t_range[1] - self.t_range[0])
                + self.t_range[0]
            )
            * self.scheduler.config.num_train_timesteps
        ).to(torch.int)
        noise = torch.randn_like(render, device=self.device)
        render_noisy = self.scheduler.add_noise(render, noise, timesteps)

        # Get positionally modulated prompts
        if phi < 60:
            pos_embeds = self.embeds["top"]
        elif theta <= 45 and theta > 315:
            pos_embeds = self.embeds["front"]
        elif theta > 135 and theta < 225:
            pos_embeds = self.embeds["side"]
        else:
            pos_embeds = self.embeds["back"]

        # Calculate loss via reparameterization
        noise_pred = self.predict_cfg_noise(
            render_noisy, timesteps, pos_embeds, self.embeds["neg"]
        )
        weight = 1 - self.alphas[timesteps].view(-1, 1, 1, 1).expand(
            -1, *render.shape[1:]
        )
        gradient = weight * (noise_pred - noise)
        sds_loss = (gradient * render).sum()

        return sds_loss

    @abstractmethod
    def predict_noise(
        self,
        render_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Abstract. Get noise prediction.

        Arguments:
            render: Render tensor of shape (N, *render_shape).
            timesteps: Timestep tensor of shape (N,).
            embeds: Text embedding tensor.

        Returns:
            Noise prediction tensor of shape (N, *render_shape).
        """
        pass

    @torch.no_grad()
    def predict_cfg_noise(
        self,
        render_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        pos_embeds: torch.Tensor,
        neg_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate classifier-free noise.

        Arguments:
            render_noisy: Noisy render tensor of shape (N, *render_shape).
            timesteps: Timestep tensor of shape (N,).
            pos_embeds: Target text embedding tensor.
            neg_embeds: Avoidance text embedding tensor.

        Returns:
            Predicted noise tensor of shape (N, *render_shape).
        """
        N, _, _, _ = render_noisy.shape

        # Predict positive and negative noise
        render_comb = torch.cat([render_noisy] * 2, dim=0)
        timesteps_comb = torch.cat([timesteps] * 2, dim=0)
        embeds_comb = torch.cat(
            [
                torch.cat([pos_embeds] * N, dim=0),
                torch.cat([neg_embeds] * N, dim=0),
            ],
            dim=0,
        )

        noise_pred_comb = self.predict_noise(render_comb, timesteps_comb, embeds_comb)
        noise_pred_pos, noise_pred_neg = noise_pred_comb.chunk(2)

        # Calculate classifier-free noise
        noise_pred = noise_pred_neg + self.guidance_scale * (
            noise_pred_pos - noise_pred_neg
        )

        return noise_pred

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """Embed text.

        Arguments:
            text: Target text.

        Returns:
            Text embedding tensor.
        """
        input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt",
        ).input_ids.to(self.device)
        text_embeds = self.text_encoder(input_ids)[0]
        return text_embeds

    @abstractmethod
    def decode_train(self, render: torch.Tensor) -> torch.Tensor:
        """Abstract. Convert render tensor to images tensor.

        Arguments:
            render: Render tensor of shape (N, *render_shape)

        Returns:
            Images tensor.
        """
        pass
