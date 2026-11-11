"""Observation validation for three cameras and four history frames."""
from pathlib import PurePosixPath
import numpy as np
LEGACY_OBSERVATION_KEYS = {'current_image_path', 'history_trajectory', 'history_time_offsets_seconds', 'velocity', 'acceleration', 'navigation_command'}
OBSERVATION_KEYS = LEGACY_OBSERVATION_KEYS | {'history_image_paths'}
CAMERA_ORDER = ('front', 'left', 'right')
CAMERA_SENSORS = {'front': 'CAM_F0', 'left': 'CAM_L1', 'right': 'CAM_R1'}

def relative_image(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute():
        for marker in ('trainval', 'test'):
            if marker in path.parts:
                path = PurePosixPath(*path.parts[path.parts.index(marker) + 1:])
                break
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe image path: {value}')
    return str(path)


def array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'Invalid {name}: shape={result.shape}, expected={shape}')
    return result


def validate_observations(obs: dict, *, multiview=True) -> None:
    if set(obs) != (OBSERVATION_KEYS if multiview else LEGACY_OBSERVATION_KEYS):
        raise ValueError('Observation field whitelist mismatch')
    if relative_image(obs['current_image_path']) != obs['current_image_path']:
        raise ValueError('Current image path must be relative')
    array(obs['history_trajectory'], (4, 3), 'history')
    array(obs['history_time_offsets_seconds'], (4,), 'history times')
    array(obs['velocity'], (2,), 'velocity')
    array(obs['acceleration'], (2,), 'acceleration')
    if obs['navigation_command'] not in (0, 1, 2, 3):
        raise ValueError('Invalid navigation')
    if multiview:
        times = array(obs['history_time_offsets_seconds'], (4,), 'history times')
        if times[-1] != 0 or times[0] < -2.05 or np.any(np.diff(times) <= 0):
            raise ValueError('History must contain four ordered real frames inside the 2-second window')
        if np.max(np.abs(times - np.round(times * 2) / 2)) > 0.05:
            raise ValueError('History timestamps are inconsistent with the nominal 2 Hz log timeline')
        views = obs['history_image_paths']
        if not isinstance(views, dict) or set(views) != set(CAMERA_ORDER):
            raise ValueError('Expected front, left and right history videos')
        for view, sensor in CAMERA_SENSORS.items():
            paths = views[view]
            if not isinstance(paths, list) or len(paths) != 4 or len(set(paths)) != 4:
                raise ValueError(f'Expected four distinct frames for {view}')
            for path in paths:
                if relative_image(path) != path or PurePosixPath(path).parts[-2] != sensor:
                    raise ValueError(f'Incorrect sensor/path for {view}: {path}')
        if views['front'][-1] != obs['current_image_path']:
            raise ValueError('Current teacher-control frame must equal the last front history frame')

