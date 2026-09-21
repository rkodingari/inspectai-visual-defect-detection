from __future__ import annotations

import numpy as np
import torch
from PIL import Image


class GradCAM:
    """Minimal Grad-CAM implementation for a selected convolutional layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activations)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, inputs: torch.Tensor, class_index: int | None = None) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        if not inputs.requires_grad:
            inputs = inputs.detach().requires_grad_(True)
        logits = self.model(inputs)
        if class_index is None:
            class_index = int(logits.argmax(dim=1)[0])
        logits[0, class_index].backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture tensors")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, inputs.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)
        return cam.cpu().numpy()

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.42) -> Image.Image:
    image = image.convert("RGB")
    # A small NumPy implementation of the familiar blue-green-yellow-red map keeps
    # the serving path independent of Matplotlib's cache and GUI backends.
    values = np.clip(heatmap, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    colored = np.stack((red, green, blue), axis=-1)
    heat = Image.fromarray((colored * 255).astype(np.uint8))
    heat = heat.resize(image.size, Image.Resampling.BILINEAR)
    return Image.blend(image, heat, alpha)
