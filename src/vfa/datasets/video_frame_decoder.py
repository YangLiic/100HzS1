"""Video decoding used by the optional LeRobot episode cache."""

from pathlib import Path

import numpy as np
import torch


def decode_video_frames_pyav_seek(video_path, timestamps, tolerance_s):
    """Seek to a preceding keyframe and convert only requested frames to RGB."""
    import av

    if not timestamps:
        raise ValueError("At least one video timestamp is required")
    timestamps = [float(t) for t in timestamps]
    closest = [None] * len(timestamps)
    distances = [float("inf")] * len(timestamps)
    with av.open(str(video_path), mode="r") as container:
        if not container.streams.video:
            raise RuntimeError(f"No video stream found in {video_path}")
        stream = container.streams.video[0]
        if stream.time_base is None:
            raise RuntimeError(f"Video stream has no time base: {video_path}")
        stream.codec_context.thread_type = "AUTO"
        stream.codec_context.thread_count = 2
        seek_pts = max(0, int(min(timestamps) / float(stream.time_base)))
        container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            time = float(frame.pts * stream.time_base)
            for i, target in enumerate(timestamps):
                distance = abs(time - target)
                if distance < distances[i]:
                    distances[i] = distance
                    closest[i] = frame
            if time >= max(timestamps):
                break
        if any(frame is None for frame in closest):
            raise RuntimeError(f"Could not decode frames for {timestamps} from {video_path}")
        if any(distance >= float(tolerance_s) for distance in distances):
            raise AssertionError(f"Decoded frames violate timestamp tolerance: "
                                 f"distances={distances}, tolerance_s={tolerance_s}, video={video_path}")
        converted = {}
        frames = []
        for frame in closest:
            if id(frame) not in converted:
                array = frame.to_ndarray(format="rgb24")
                converted[id(frame)] = torch.from_numpy(array).permute(2, 0, 1).to(torch.float32).div_(255)
            frames.append(converted[id(frame)])
    return torch.stack(frames)


def decode_all_video_frames_pyav(video_path: str | Path, expected_num_frames=None):
    """Sequential decode to uint8 NCHW, keeping the shared footprint small."""
    import av

    arrays = []
    with av.open(str(video_path), mode="r") as container:
        if not container.streams.video:
            raise RuntimeError(f"No video stream found in {video_path}")
        stream = container.streams.video[0]
        stream.codec_context.thread_type = "AUTO"
        stream.codec_context.thread_count = 2
        for frame in container.decode(stream):
            arrays.append(frame.to_ndarray(format="rgb24"))
    if expected_num_frames is not None and len(arrays) != expected_num_frames:
        raise RuntimeError(f"Decoded {len(arrays)} frames from {video_path}, "
                           f"expected {expected_num_frames} (one per episode row)")
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
