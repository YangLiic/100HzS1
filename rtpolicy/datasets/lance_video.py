"""CPU and NVDEC video decoding with explicit color-range handling."""

from collections import OrderedDict
import io
import math
import os


class MountedFilePool:
    """Per-worker LRU of file descriptors; never caches file contents."""

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("Mounted file capacity must be positive")
        self.capacity = capacity
        self._files = OrderedDict()

    def readinto(self, path, offset, buffer):
        if path not in self._files:
            if len(self._files) >= self.capacity:
                _, old = self._files.popitem(last=False)
                os.close(old)
            self._files[path] = os.open(path, os.O_RDONLY)
        self._files.move_to_end(path)
        fd = self._files[path]
        if hasattr(os, "preadv"):
            return os.preadv(fd, [buffer], offset)
        data = os.pread(fd, len(buffer), offset)
        buffer[:len(data)] = data
        return len(data)

    def close(self):
        for fd in self._files.values():
            os.close(fd)
        self._files.clear()


class FileSlice(io.RawIOBase):
    """Seekable, bounded view of an uncompressed blob in a mounted Lance file.

    Only requested ranges are read. A shared descriptor pool avoids reopening
    one physical Lance file for each logical video. Both caches are bounded.
    """

    def __init__(self, path, offset, size, pool=None):
        super().__init__()
        self._file = None
        if offset < 0 or size < 0:
            raise ValueError("File slice offset and size must be nonnegative")
        self._path = path
        self._pool = pool
        if pool is None:
            self._file = io.FileIO(path, "r")
        self._offset = offset
        self._size = size
        self._position = 0
        if self._file is not None:
            self._file.seek(offset)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        self._checkClosed()
        return self._position

    def seek(self, offset, whence=io.SEEK_SET):
        self._checkClosed()
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        if position < 0:
            raise ValueError("Negative file slice seek position")
        if self._file is not None:
            self._file.seek(self._offset + position)
        self._position = position
        return position

    def readinto(self, buffer):
        self._checkClosed()
        size = min(len(buffer), max(0, self._size - self._position))
        if not size:
            return 0
        target = memoryview(buffer)[:size]
        if self._pool is None:
            count = self._file.readinto(target)
        else:
            count = self._pool.readinto(self._path, self._offset + self._position, target)
        self._position += count
        return count

    def close(self):
        if not self.closed and self._file is not None:
            self._file.close()
        super().close()


def mounted_video_slices(dataset, table_uri):
    """Resolve blob v1 descriptors to mounted file slices, without reading MP4s.

    This intentionally supports only single-file fragments with the public v1
    (position,size) descriptor schema. Other layouts use Lance's BlobFile API.
    """
    from pathlib import Path

    root = Path(table_uri)
    if not root.is_dir():
        return None
    descriptions = dataset.to_table(
        columns=["video_key", "chunk_index", "file_index", "video_bytes"],
        blob_handling="blobs_descriptions", with_row_address=True, scan_in_order=True,
    )
    descriptor_type = descriptions.schema.field("video_bytes").type
    if not hasattr(descriptor_type, "names") or descriptor_type.names != ["position", "size"]:
        return None
    paths = {}
    for fragment in dataset.get_fragments():
        metadata = fragment.metadata
        if len(metadata.files) != 1:
            return None
        if getattr(metadata.files[0], "base_id", None) is not None:
            return None
        path = (root / "data" / metadata.files[0].path).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            return None
        paths[fragment.fragment_id] = (str(path), path.stat().st_size)
    result = {}
    for row in descriptions.to_pylist():
        descriptor = row["video_bytes"]
        if descriptor is None:
            return None
        path, file_size = paths[int(row["_rowaddr"]) >> 32]
        offset, size = int(descriptor["position"]), int(descriptor["size"])
        if offset < 0 or size < 0 or offset + size > file_size:
            return None
        identity = (row["video_key"], int(row["chunk_index"]), int(row["file_index"]))
        result[identity] = (path, offset, size)
    return result


class _BlobReader(io.RawIOBase):
    """Adapt Lance 7's bytearray-only readinto for BufferedReader memoryviews."""

    def __init__(self, blob):
        self.blob = blob

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.blob.tell()

    def seek(self, offset, whence=io.SEEK_SET):
        return self.blob.seek(offset, whence)

    def readinto(self, buffer):
        if isinstance(buffer, bytearray):
            return self.blob.readinto(buffer)
        data = self.blob.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def close(self):
        if not self.closed:
            self.blob.close()
        super().close()


def buffered_video_blob(blob, buffer_size=1024 * 1024):
    return io.BufferedReader(_BlobReader(blob), buffer_size=buffer_size)


class VideoDecoder:
    """Read RGB frames by presentation timestamp without materializing an MP4."""

    def __init__(self, source, tolerance_s=1e-4, seek_gap_s=1.0):
        import av

        self.source = source
        self.container = av.open(source, mode="r")
        self.stream = self.container.streams.video[0]
        # Every DataLoader worker owns its decoder; avoid FFmpeg thread pools
        # multiplying the worker count.
        self.stream.thread_count = 1
        self.tolerance_s = tolerance_s
        self.seek_gap_s = seek_gap_s

    def decode(self, timestamps):
        """Return HWC uint8 arrays, preserving query order and duplicates."""
        if not timestamps:
            return []
        unique = sorted(set(float(t) for t in timestamps))
        frames = {}
        iterator = None
        previous = None
        previous_pts = None
        previous_frame = None
        for target in unique:
            if not math.isfinite(target) or target < -self.tolerance_s:
                raise ValueError(f"Invalid video timestamp: {target}")
            if previous_pts is not None and abs(previous_pts - target) <= self.tolerance_s:
                frames[target] = previous_frame
                previous = target
                continue
            if iterator is None or target - previous > self.seek_gap_s:
                offset = max(0, math.floor((target - self.tolerance_s) / float(self.stream.time_base)))
                self.container.seek(offset, stream=self.stream, backward=True, any_frame=False)
                iterator = self.container.decode(self.stream)
            for frame in iterator:
                if frame.pts is None:
                    continue
                pts = float(frame.pts * frame.time_base)
                if pts < target - self.tolerance_s:
                    continue
                if abs(pts - target) > self.tolerance_s:
                    raise ValueError(
                        f"Video timestamp {target:.6f}s has no frame within "
                        f"{self.tolerance_s}s (next frame: {pts:.6f}s)"
                    )
                frames[target] = self._convert_frame(frame)
                previous_pts, previous_frame = pts, frames[target]
                break
            else:
                raise ValueError(f"Video ended before timestamp {target:.6f}s")
            previous = target
        return [frames[float(t)] for t in timestamps]

    def _convert_frame(self, frame):
        return frame.to_ndarray(format="rgb24")

    def close(self):
        self.container.close()
        if hasattr(self.source, "close"):
            self.source.close()


class CudaVideoDecoder(VideoDecoder):
    """NVDEC to GPU NV12/P010 planes, then batched YCbCr-to-RGB on GPU.

    PyAV 17.1 exports hardware planes through DLPack. Decode and color
    conversion both stay on CUDA, and honor the source's range/matrix.
    """

    def __init__(self, source, tolerance_s=1e-4, device="cuda:0", seek_gap_s=0.25):
        import av
        import torch
        from av.codec.hwaccel import HWAccel

        if tuple(map(int, av.__version__.split(".")[:2])) < (17, 1):
            raise RuntimeError("NVDEC video decoding requires PyAV >=17.1; install the lance-video extra")
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        # Reuse the active PyTorch context. FFmpeg's "primary context" mode
        # otherwise requests scheduling flags incompatible with an existing
        # model/transfer CUDA context.
        with torch.cuda.device(self.device):
            torch.empty(0, device=self.device)
            hardware = HWAccel(
                "cuda", device=str(self.device.index), allow_software_fallback=False,
                is_hw_owned=True, options={"primary_ctx": "0", "current_ctx": "1"},
            )
            self.source = source
            self.container = av.open(source, mode="r", hwaccel=hardware)
        self.stream = self.container.streams.video[0]
        self.stream.thread_count = 1
        self.tolerance_s = tolerance_s
        self.seek_gap_s = seek_gap_s
        self._pending_planes = None

    def _convert_frame(self, frame):
        import torch

        if frame.format.name != "cuda" or len(frame.planes) != 2:
            raise RuntimeError("NVDEC requires CUDA NV12/P010 frames; software fallback is disabled")
        # PyAV's DLPack export currently has no producer-stream handshake.
        # Synchronize before export, and keep borrowed planes alive until
        # their copies have completed. Only selected observation frames copy.
        torch.cuda.synchronize(self.device)
        self._pending_planes = None
        y = torch.from_dlpack(frame.planes[0].__dlpack__())
        uv = torch.from_dlpack(frame.planes[1].__dlpack__())
        self._pending_planes = (y, uv)
        return (y.clone(), uv.clone(), int(frame.colorspace), int(frame.color_range))

    def decode(self, timestamps):
        import torch
        import torch.nn.functional as F

        try:
            raw = super().decode(timestamps)
        finally:
            if self._pending_planes is not None:
                torch.cuda.synchronize(self.device)
                self._pending_planes = None
        if not raw:
            return []
        unique = list({id(frame): frame for frame in raw}.values())
        positions = {id(frame): pos for pos, frame in enumerate(unique)}
        y = torch.stack([frame[0] for frame in unique]).float().unsqueeze(1)
        uv = torch.stack([frame[1] for frame in unique]).float().permute(0, 3, 1, 2)
        if unique[0][0].dtype == torch.uint16:
            # P010/P016 store luma/chroma in the high bits of 16-bit words.
            y.div_(256.0)
            uv.div_(256.0)
        uv = F.interpolate(uv, size=y.shape[-2:], mode="nearest")
        colorspace, color_range = unique[0][2:]
        if any(frame[2:] != (colorspace, color_range) for frame in unique):
            raise ValueError("Video color metadata changes within a batch")
        if colorspace not in (1, 2, 4, 5, 6, 7, 9) or color_range not in (0, 1, 2):
            raise ValueError(f"Unsupported NVDEC color metadata: matrix={colorspace}, range={color_range}; use CPU decoding")
        if color_range == 2:  # AVCOL_RANGE_JPEG (full range)
            cb, cr = uv[:, 0:1] - 128, uv[:, 1:2] - 128
        else:  # MPEG/unspecified: studio range
            y.sub_(16).mul_(255 / 219)
            cb, cr = (uv[:, 0:1] - 128) * (255 / 224), (uv[:, 1:2] - 128) * (255 / 224)
        # FFmpeg defaults unspecified SD color space to BT.601.
        matrices = {
            1: (0.2126, 0.0722), 4: (0.30, 0.11),
            7: (0.212, 0.087), 9: (0.2627, 0.0593),
        }
        kr, kb = matrices.get(colorspace, (0.299, 0.114))
        kg = 1 - kr - kb
        r = y + 2 * (1 - kr) * cr
        b = y + 2 * (1 - kb) * cb
        g = y - 2 * kb * (1 - kb) / kg * cb - 2 * kr * (1 - kr) / kg * cr
        rgb = torch.cat([r, g, b], dim=1).floor_().clamp_(0, 255).to(torch.uint8)
        return [rgb[positions[id(frame)]] for frame in raw]


class TorchCodecVideoDecoder:
    """Optional batched CPU decoder with checked presentation timestamps."""

    def __init__(self, source, tolerance_s=1e-4, device="cpu", seek_mode="approximate"):
        if str(device).startswith("cuda"):
            raise ValueError("Use video_backend=nvdec for CUDA; TorchCodec 0.5 has incompatible NV12 color conversion")
        try:
            from torchcodec.decoders import VideoDecoder as NativeVideoDecoder
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "TorchCodec CPU video decoding requires a build matching PyTorch; see docs/README.md."
            ) from exc
        self.source = source
        self.tolerance_s = tolerance_s
        self.device = device
        try:
            self.decoder = NativeVideoDecoder(
                source, device=device, dimension_order="NCHW",
                num_ffmpeg_threads=1, seek_mode=seek_mode,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Cannot open TorchCodec video decoder on {device}: {exc}. "
                "Check the TorchCodec/PyTorch/FFmpeg versions. See docs/README.md."
            ) from exc
        self.fps = self.decoder.metadata.average_fps
        self.begin = self.decoder.metadata.begin_stream_seconds
        if self.fps is None or self.fps <= 0:
            self.close()
            raise ValueError("Video metadata has no valid average FPS")

    def decode(self, timestamps):
        if not timestamps:
            return []
        indices = []
        for ts in timestamps:
            if not math.isfinite(ts) or ts < self.begin - self.tolerance_s:
                raise ValueError(f"Invalid video timestamp: {ts}")
            indices.append(round((ts - self.begin) * self.fps))
        unique = sorted(set(indices))
        # LeRobot uses constant-FPS MP4s. Approximate seeking avoids an initial
        # whole-file scan; check actual PTS so a mismatched/VFR file cannot
        # silently yield a different frame.
        batch = self.decoder.get_frames_at(unique)
        positions = {idx: pos for pos, idx in enumerate(unique)}
        actual_pts = batch.pts_seconds.tolist()
        for ts, idx in zip(timestamps, indices):
            pts = actual_pts[positions[idx]]
            if abs(pts - ts) > self.tolerance_s:
                raise ValueError(
                    f"Video timestamp {ts:.6f}s has no frame within "
                    f"{self.tolerance_s}s (decoded frame: {pts:.6f}s)"
                )
        return [batch.data[positions[idx]] for idx in indices]

    def close(self):
        # Destroy the decoder before closing its callback source.
        self.decoder = None
        if hasattr(self.source, "close"):
            self.source.close()


class VideoDecoderCache:
    """Bounded LRU of open decoders, keyed by camera and source file."""

    def __init__(
        self, capacity=16, tolerance_s=1e-4, device="cpu", backend="auto",
        seek_mode="approximate", seek_gap_s=None,
    ):
        if capacity < 1:
            raise ValueError("Video decoder cache capacity must be positive")
        if tolerance_s <= 0:
            raise ValueError("Video timestamp tolerance must be positive")
        self.capacity = capacity
        self.tolerance_s = tolerance_s
        self.device = device or "cpu"
        if backend not in ("auto", "pyav", "torchcodec", "nvdec"):
            raise ValueError(f"Unknown video backend: {backend}")
        cuda = str(self.device).startswith("cuda")
        self.backend = ("nvdec" if cuda else "pyav") if backend == "auto" else backend
        if self.backend == "pyav" and cuda:
            self.backend = "nvdec"
        if self.backend == "nvdec" and not cuda:
            raise ValueError("video_backend=nvdec requires decode_device=cuda")
        if self.backend == "torchcodec" and cuda:
            raise ValueError("Use video_backend=nvdec for CUDA; TorchCodec 0.5 has incompatible NV12 color conversion")
        if seek_mode not in ("exact", "approximate"):
            raise ValueError(f"Unknown video seek mode: {seek_mode}")
        self.seek_mode = seek_mode
        self.seek_gap_s = (0.25 if cuda else 1.0) if seek_gap_s is None else seek_gap_s
        if self.seek_gap_s < 0:
            raise ValueError("Video seek gap must be nonnegative")
        self.decoders = OrderedDict()

    def get(self, key, source_factory):
        if key in self.decoders:
            self.decoders.move_to_end(key)
            return self.decoders[key]
        source = source_factory()
        try:
            if self.backend == "nvdec":
                decoder = CudaVideoDecoder(source, self.tolerance_s, self.device, self.seek_gap_s)
            elif self.backend == "torchcodec":
                decoder = TorchCodecVideoDecoder(source, self.tolerance_s, self.device, self.seek_mode)
            else:
                decoder = VideoDecoder(source, self.tolerance_s, self.seek_gap_s)
        except Exception:
            if hasattr(source, "close"):
                source.close()
            raise
        self.decoders[key] = decoder
        while len(self.decoders) > self.capacity:
            _, old = self.decoders.popitem(last=False)
            old.close()
        return decoder

    def close(self):
        for decoder in self.decoders.values():
            decoder.close()
        self.decoders.clear()
