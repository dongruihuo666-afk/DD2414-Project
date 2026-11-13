"""Small Python integration surface; no weights, downloads or global model state."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureOutput:
    """Existing model features, not segmentation logits or a depth GT map."""
    bev_features: Any
    image_features: Any
    depth_prediction: Any
    location: str


def extract_features(model, inputs, *, location='fused_bev'):
    """Read an existing MultiSweepTeacher/MetricDepthTeacher feature location.

    This does not freeze/train the caller's model, move tensors, or enter
    no_grad/autocast. The caller controls those policies. See docs/INTERFACES.md.
    """
    if location not in ('camera_bev', 'fused_bev'):
        raise ValueError('location must be camera_bev or fused_bev')
    bev, image, depth = model.features(inputs, location=location)
    return FeatureOutput(bev, image, depth, location)
