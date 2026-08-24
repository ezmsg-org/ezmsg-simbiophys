"""MLX implementation detail for :mod:`ezmsg.simbiophys.cosine_encoder`.

Kept separate so importing ezmsg-simbiophys never requires MLX. The public
transformer imports this module lazily only after receiving an MLX array.
"""

import mlx.core as mx


def _cosine_encode(polar, baseline, modulation, preferred_direction, speed_modulation):
    magnitude = polar[:, 0:1]
    angle = polar[:, 1:2]
    return baseline + modulation * magnitude * mx.cos(angle - preferred_direction) + speed_modulation * magnitude


cosine_encode = mx.compile(_cosine_encode)
"""Shape-specialized compiled cosine encoder shared by transformer instances."""
