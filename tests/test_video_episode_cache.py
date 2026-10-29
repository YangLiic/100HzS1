"""Exercise real LeRobot tables, video decode, shared ranks and worker startup."""
from importlib import import_module as _import_module

from copy import deepcopy
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
close_managed_episode_caches = _import_module('100hzs1.datasets.managed_episode_cache').close_managed_episode_caches
apply_bgr_to_rgb = _import_module('100hzs1.datasets.image_color').apply_bgr_to_rgb
S1Dataset = _import_module('100hzs1.datasets.dataset').S1Dataset
decode_all_video_frames_pyav = _import_module('100hzs1.datasets.video_frame_decoder').decode_all_video_frames_pyav
decode_video_frames_pyav_seek = _import_module('100hzs1.datasets.video_frame_decoder').decode_video_frames_pyav_seek


def _legacy_frames(dataset, queries, episode):
    return {key: decode_video_frames_pyav_seek(
        dataset.root / dataset.meta.get_video_file_path(episode, key), timestamps, dataset.tolerance_s,
    ).squeeze(0) for key, timestamps in queries.items()}


def _make_dataset(root, **overrides):
    args = dict(repo_id="test/video-cache", root=root,
                dataset_metadata=LeRobotDatasetMetadata("test/video-cache", root=root),
                horizon=4, n_obs_steps=2, obs_image_frame_indices=[0, 1],
                rgb_key=["head_cam"], pad_before=2, val_ratio=0,
                cache_to_ram=True, cache_backend="shm", cache_build_num_workers=2)
    args.update(overrides)
    return S1Dataset(**args)


def _rank_reader(root, rendezvous, output, release):
    os.environ.update(WORLD_SIZE="2", MASTER_ADDR="127.0.0.1", MASTER_PORT=rendezvous)
    try:
        dataset = _make_dataset(root)
        item = dataset[3]
        client = dataset._episode_cache_backend
        output.put((dataset.cache_socket, os.fstat(client.fd).st_ino,
                    item["obs"]["head_cam"].sum().item()))
        release.wait(60)
    finally:
        close_managed_episode_caches()


def _write_fixture(root):
    (root / "meta").mkdir(parents=True)
    features = {
        "state": {"dtype": "float32", "shape": [2], "names": ["x", "y"]},
        "action": {"dtype": "float32", "shape": [2], "names": ["x", "y"]},
        **{key: {"dtype": "video", "shape": [3, 12, 16], "names": ["channels", "height", "width"],
                 "info": {"video.fps": 10, "video.height": 12, "video.width": 16,
                          "video.channels": 3, "video.codec": "ffv1", "video.pix_fmt": "bgr0"}}
           for key in ["head_cam", "wrist_cam"]},
        **{key: {"dtype": "int64", "shape": [1], "names": None}
           for key in ["frame_index", "episode_index", "index", "task_index"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    }
    info = dict(codebase_version="v2.1", robot_type="test", total_episodes=2, total_frames=12,
                total_tasks=1, total_videos=4, total_chunks=1, chunks_size=1000, fps=10,
                splits={"train": "0:2"}, features=features,
                data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mkv")
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "test"}) + "\n")
    episodes = [{"episode_index": ep, "tasks": ["test"], "length": 6} for ep in range(2)]
    (root / "meta/episodes.jsonl").write_text("".join(json.dumps(ep) + "\n" for ep in episodes))
    stats = {"state": {"min": [0, 0], "max": [11, 12], "mean": [5, 6], "std": [1, 1], "count": [6]}}
    (root / "meta/episodes_stats.jsonl").write_text("".join(
        json.dumps({"episode_index": ep, "stats": stats}) + "\n" for ep in range(2)))
    for ep in range(2):
        data = {"state": [[float(ep * 6 + i), float(i + 1)] for i in range(6)],
                "action": [[float(i + 2), float(i + 3)] for i in range(6)],
                "timestamp": np.arange(6, dtype=np.float32) / 10,
                "frame_index": list(range(6)), "episode_index": [ep] * 6,
                "index": list(range(ep * 6, ep * 6 + 6)), "task_index": [0] * 6}
        table = root / f"data/chunk-000/episode_{ep:06d}.parquet"
        table.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pydict(data), table)
        for camera, offset in [("head_cam", 0), ("wrist_cam", 40)]:
            video = root / f"videos/chunk-000/{camera}/episode_{ep:06d}.mkv"
            video.parent.mkdir(parents=True, exist_ok=True)
            with av.open(str(video), "w") as container:
                stream = container.add_stream("ffv1", rate=10)
                stream.width, stream.height, stream.pix_fmt = 16, 12, "bgr0"
                for i in range(6):
                    value = ep * 60 + i * 5 + offset
                    array = np.empty((12, 16, 3), dtype=np.uint8)
                    array[:] = [value, value + 10, value + 20]
                    for packet in stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)


class VideoEpisodeCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory(prefix="s1_video_test_")
        cls.root = Path(cls.directory.name) / "dataset"
        _write_fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        close_managed_episode_caches()
        cls.directory.cleanup()

    def test_cache_matches_baseline_at_episode_edges_and_preserves_float_history(self):
        baseline = _make_dataset(self.root, cache_to_ram=False)
        cached = _make_dataset(self.root)
        self.assertIsNone(baseline._episode_cache)
        with patch.object(LeRobotDataset, "_query_videos", _legacy_frames), patch(
            "100hzs1.datasets.dataset.decode_video_frames_pyav_seek", side_effect=AssertionError("cache miss")
        ):
            for idx in [0, 1, 3, 5, 6, 7, 11]:
                expected, actual = baseline[idx], cached[idx]
                for section in expected:
                    self.assertEqual(expected[section].keys(), actual[section].keys())
                    for key in expected[section]:
                        torch.testing.assert_close(actual[section][key], expected[section][key], rtol=0, atol=0)
                self.assertEqual(actual["obs"]["head_cam"].shape, (2, 3, 12, 16))
                self.assertEqual(actual["obs"]["head_cam"].dtype, torch.float32)
                self.assertNotIn("wrist_cam", actual["obs"])
        metadata = cached._episode_cache_backend.metadata["columns"]
        self.assertEqual(metadata["head_cam"]["dtype"], "uint8")
        self.assertNotIn("wrist_cam", metadata)

    def test_master_switch_calls_original_decoder_and_creates_no_service(self):
        with patch("100hzs1.datasets.managed_episode_cache.ensure_episode_cache", side_effect=AssertionError("started cache")):
            dataset = _make_dataset(self.root, cache_to_ram=False)
        with patch.object(LeRobotDataset, "_query_videos", autospec=True, side_effect=_legacy_frames) as original:
            dataset[3]
            original.assert_called_once()

    def test_timestamp_queries_do_not_construct_hf_selections_on_cache_hit(self):
        dataset = _make_dataset(self.root)
        with patch.object(dataset.hf_dataset, "select", side_effect=AssertionError("HF selection")):
            dataset[3]

    def test_partial_cache_decodes_only_missed_episode(self):
        dataset = _make_dataset(self.root, cache_max_episodes=1)
        with patch("100hzs1.datasets.dataset.decode_video_frames_pyav_seek", wraps=decode_video_frames_pyav_seek) as decode:
            dataset[3]
            self.assertEqual(decode.call_count, 0)
            dataset[9]
            self.assertEqual(decode.call_count, 1)

    def test_scalar_only_cache_keeps_optimized_seek(self):
        dataset = _make_dataset(self.root, cache_decoded_video=False)
        with patch("100hzs1.datasets.dataset.decode_video_frames_pyav_seek", wraps=decode_video_frames_pyav_seek) as decode:
            dataset[3]
            self.assertEqual(decode.call_count, 1)
        self.assertNotIn("head_cam", dataset._episode_cache_backend.metadata["columns"])

    def test_attached_client_can_be_pickled_for_spawn_workers(self):
        dataset = _make_dataset(self.root)
        dataset[3]
        clone = pickle.loads(pickle.dumps(dataset._episode_cache_backend))
        self.assertIsNone(clone.mm)
        frames = clone.get_rows({"head_cam": [0, 1]})
        self.assertEqual(len(frames["head_cam"]), 2)
        del frames
        clone.close()

    def test_spawn_dataloader_reads_an_already_attached_dataset(self):
        dataset = _make_dataset(self.root, bgr_to_rgb=True)
        expected = dataset[0]["obs"]["head_cam"]
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=2,
                                            multiprocessing_context="spawn")
        batch = next(iter(loader))
        torch.testing.assert_close(batch["obs"]["head_cam"][0], expected)

    def test_two_training_ranks_share_one_memfd(self):
        context = mp.get_context("spawn")
        output, release = context.Queue(), context.Event()
        processes = [context.Process(target=_rank_reader, args=(self.root, str(os.getpid()), output, release))
                     for _ in range(2)]
        for process in processes:
            process.start()
        try:
            results = [output.get(timeout=60) for _ in processes]
            self.assertEqual(results[0], results[1])
        finally:
            release.set()
            for process in processes:
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
                    process.join()
        self.assertEqual([p.exitcode for p in processes], [0, 0])
        self.assertFalse(Path(results[0][0]).exists())

    def test_validation_has_no_cache(self):
        dataset = _make_dataset(self.root, val_ratio=0.5, bgr_to_rgb=True)
        validation = dataset.get_validation_dataset()
        self.assertTrue(validation.bgr_to_rgb)
        self.assertFalse(validation.cache_to_ram)
        self.assertIsNone(validation._episode_cache)

    def test_bgr_correction_is_independent_of_all_video_cache_modes(self):
        raw_dataset = _make_dataset(self.root, cache_to_ram=False)
        with patch.object(LeRobotDataset, "_query_videos", _legacy_frames):
            expected = raw_dataset[3]["obs"]["head_cam"].flip(-3)
            self.assertFalse(torch.equal(expected[:, 0], expected[:, 2]))
            for cache, decoded in [(False, True), (False, False), (True, False), (True, True)]:
                with self.subTest(cache_to_ram=cache, cache_decoded_video=decoded):
                    dataset = _make_dataset(self.root, cache_to_ram=cache,
                                            cache_decoded_video=decoded, bgr_to_rgb=True)
                    actual = dataset[3]["obs"]["head_cam"]
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(dataset[3]["obs"]["head_cam"], expected, rtol=0, atol=0)

    def test_bgr_correction_precedes_transforms_and_keeps_shared_source_unchanged(self):
        dataset = _make_dataset(self.root, bgr_to_rgb=True)
        client = dataset._episode_cache_backend
        original = client.get_rows({"head_cam": [1]})["head_cam"][0].clone()
        seen = []
        dataset.image_transforms = lambda image: (seen.append(image.clone()) or image)
        item = dataset[3]
        expected = original.flip(0).float() / 255
        torch.testing.assert_close(seen[0][0], expected, rtol=0, atol=0)
        torch.testing.assert_close(item["obs"]["head_cam"][0], expected, rtol=0, atol=0)
        torch.testing.assert_close(client.get_rows({"head_cam": [1]})["head_cam"][0], original, rtol=0, atol=0)

    def test_bgr_correction_survives_partial_cache_misses(self):
        raw = _make_dataset(self.root, cache_to_ram=False)
        dataset = _make_dataset(self.root, bgr_to_rgb=True, cache_max_episodes=1)
        with patch.object(LeRobotDataset, "_query_videos", _legacy_frames):
            for idx in [3, 9]:
                expected = raw[idx]["obs"]["head_cam"].flip(-3)
                torch.testing.assert_close(dataset[idx]["obs"]["head_cam"], expected, rtol=0, atol=0)

    def test_full_decode_and_seek_return_same_pixels_in_query_order(self):
        video = self.root / "videos/chunk-000/head_cam/episode_000000.mkv"
        full = decode_all_video_frames_pyav(video, expected_num_frames=6)
        sought = decode_video_frames_pyav_seek(video, [0.4, 0.2, 0.2], tolerance_s=1e-4)
        torch.testing.assert_close(sought, full[[4, 2, 2]].float() / 255, rtol=0, atol=0)


class ImageColorTests(unittest.TestCase):
    def test_channel_axis_dtype_and_source_preservation(self):
        for dtype in [torch.uint8, torch.float32]:
            for shape in [(3, 2, 4), (2, 3, 2, 4)]:
                source = torch.arange(int(np.prod(shape))).reshape(shape).to(dtype)
                before = source.clone()
                item = {"camera": source, "state": torch.tensor([1., 2.])}
                apply_bgr_to_rgb(item, ["camera", "absent_camera"])
                expected = source.index_select(-3, torch.tensor([2, 1, 0]))
                torch.testing.assert_close(item["camera"], expected)
                torch.testing.assert_close(source, before)
                self.assertEqual(item["camera"].dtype, dtype)
                torch.testing.assert_close(item["state"], torch.tensor([1., 2.]))

    def test_invalid_image_layout_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "CHW/TCHW"):
            apply_bgr_to_rgb({"camera": torch.zeros(4, 5, 3)}, ["camera"])

    def test_lance_postprocessing_uses_the_same_color_correction(self):
        try:
            LanceDataset = _import_module('100hzs1.datasets.lance_dataset').S1Dataset
        except ImportError:
            self.skipTest("optional Lance dependencies are unavailable")
        from types import SimpleNamespace
        dataset = LanceDataset.__new__(LanceDataset)
        dataset.bgr_to_rgb = True
        dataset.meta = SimpleNamespace(camera_keys=["camera"])
        dataset._organize_item_data = lambda item: item
        source = torch.arange(24).reshape(1, 3, 2, 4).float()
        result = dataset._postprocess_item({"camera": source})
        torch.testing.assert_close(result["camera"], source.flip(-3))


if __name__ == "__main__":
    unittest.main()
