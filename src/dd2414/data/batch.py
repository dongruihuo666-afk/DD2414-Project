"""Named, zero-copy views of the unchanged legacy NuScenesData tuple.

Construction performs no device/dtype conversion, augmentation or validation.
Use validate() explicitly at an integration boundary, not inside a hot loop.
"""
from dataclasses import dataclass
from typing import Any, ClassVar, Sequence


@dataclass(frozen=True)
class NuScenesBatch:
    values: Sequence[Any]
    FIELDS: ClassVar[tuple[str, ...]] = (
        'images', 'camera_rotations', 'camera_translations', 'intrinsics',
        'lidar_current', 'lidar_current_extra', 'lidar', 'lidar_extra',
        'boxes', 'visibility', 'track_ids', 'scores', 'vehicle_mask',
        'valid_mask', 'centers', 'offsets', 'radar', 'ego_pose',
    )

    def __getattr__(self, name):
        if name in self.FIELDS:
            return self.values[self.FIELDS.index(name)]
        raise AttributeError(name)

    def validate(self):
        """Check the public full-tuple shape contract without changing values.

        Padded point count, object count and lidar feature dimensions vary.
        Value/range checks belong to dataset/source admission, not this adapter.
        """
        if len(self.values) != len(self.FIELDS):
            raise ValueError('Expected the full 18-field NuScenesData batch')
        if self.images.ndim != 6 or self.images.shape[3] != 3:
            raise ValueError('images must be [B,T,S,3,H,W]')
        b, t, s = self.images.shape[:3]
        for name, tail in [('camera_rotations', (3, 3)),
                           ('camera_translations', (3,)), ('intrinsics', (4, 4))]:
            if tuple(getattr(self, name).shape) != (b, t, s, *tail):
                raise ValueError(name + ' does not match image cameras')
        if self.radar.ndim != 4 or tuple(self.radar.shape[:3]) != (b, t, 19):
            raise ValueError('radar must be [B,T,19,R]')
        for name, channels in [('vehicle_mask', 1), ('valid_mask', 1),
                               ('centers', 1), ('offsets', 2)]:
            value = getattr(self, name)
            if value.ndim != 5 or tuple(value.shape[:3]) != (b, t, channels):
                raise ValueError(name + ' must be [B,T,C,Z,X]')
        return self


def as_batch(values):
    """Return an existing view or wrap a legacy sequence without copying it."""
    return values if isinstance(values, NuScenesBatch) else NuScenesBatch(values)
