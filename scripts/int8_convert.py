#!/usr/bin/env python3
"""INT8 PTQ (entropy calibration) for RTPolicy policy ONNX — cluster-side graph work.

Same persistent-workdir patches as fp16_convert.py.

Usage: python3 int8_convert.py <in.onnx> <out.onnx> <calib_dir> [num_samples]
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import onnx
from google.protobuf.message import EncodeError
from modelopt.onnx import utils as onnx_utils

WORK_DIR = None


def _patch_byte_size():
    def byte_size_patched(self, **kwargs):
        total = 0
        for tensor in self.graph.initializer:
            total += len(tensor.raw_data) if tensor.raw_data else 8
        return max(total * 2, 3 * (1024**3))
    onnx.ModelProto.ByteSize = byte_size_patched


def _patch_save_onnx():
    def save_onnx_patched(model, onnx_path, save_as_external_data=False):
        try:
            proto = model.SerializeToString()
            save_as_external_data = save_as_external_data or len(proto) > 2 * (1024**3)
        except (ValueError, EncodeError):
            save_as_external_data = True
        model.ir_version = 10
        if save_as_external_data:
            data_name = os.path.basename(onnx_path) + ".data"
            # Write graph+data to WORK_DIR, then move BOTH next to the target
            # path so later onnx.load(onnx_path) finds the data.
            tmp_graph = os.path.join(WORK_DIR, os.path.basename(onnx_path))
            data_path = os.path.join(os.path.dirname(os.path.abspath(onnx_path)), data_name)
            if os.path.exists(data_path):
                os.remove(data_path)
            onnx.save_model(
                model,
                tmp_graph,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=data_name,
                size_threshold=1024,
            )
            shutil.move(tmp_graph, onnx_path)
            if os.path.exists(os.path.join(WORK_DIR, data_name)):
                shutil.move(os.path.join(WORK_DIR, data_name), data_path)
            # else: tensors stayed external refs at the ORIGINAL input .data —
            # nothing to move; the final materialize step rewrites them.
        else:
            onnx.save(model, onnx_path)

    # Patch the utils module AND every already-imported reference
    # (quantize.py does `from ..utils import save_onnx`, binding its own name).
    onnx_utils.save_onnx = save_onnx_patched
    import importlib
    for _mod_name in (
        "modelopt.onnx.quantization.quantize",
        "modelopt.onnx.trt_utils",
        "modelopt.onnx.quantization.int8",
        "modelopt.onnx.quantization.int8.quantize_int8",
        "modelopt.onnx.quantization.int4",
        "modelopt.onnx.quantization.graph_utils",
    ):
        try:
            _m = importlib.import_module(_mod_name)
            if hasattr(_m, "save_onnx"):
                _m.save_onnx = save_onnx_patched
        except ImportError:
            pass


def _patch_infer_shapes():
    def infer_shapes_patched(model, **kwargs):
        orig_path = os.path.join(WORK_DIR, "m.onnx")
        inferred_path = os.path.join(WORK_DIR, "i.onnx")
        onnx_utils.save_onnx(model, orig_path, save_as_external_data=True)
        onnx.shape_inference.infer_shapes_path(orig_path, inferred_path, **kwargs)
        return onnx.load(inferred_path, load_external_data=True)
    onnx_utils.infer_shapes = infer_shapes_patched


def _patch_check_model():
    def check_model_patched(model):
        path = os.path.join(WORK_DIR, "c.onnx")
        onnx_utils.save_onnx(model, path, save_as_external_data=True)
        onnx.checker.check_model(path)
        return onnx.load(path, load_external_data=True)
    onnx_utils.check_model = check_model_patched


def load_calibration_dict(calib_dir, input_names, max_samples):
    # Layout A (ar5s1): calib_dir/sample_XXX/inputs/<name>.npy
    # Layout B (real_cover_0901): calib_dir/<name>/sample_NNN.npy  (hb_compile layout)
    stacks = {name: [] for name in input_names}
    if os.path.isdir(os.path.join(calib_dir, "inputs")) or os.path.isdir(
        os.path.join(calib_dir, "sample_000")
    ):
        sample_dirs = sorted(
            d for d in os.listdir(calib_dir)
            if d.startswith("sample_") and os.path.isdir(os.path.join(calib_dir, d, "inputs"))
        )[:max_samples]
        for sd in sample_dirs:
            for name in input_names:
                path = os.path.join(calib_dir, sd, "inputs", name + ".npy")
                if os.path.isfile(path):
                    stacks[name].append(np.load(path))
        n_samples = len(sample_dirs)
    else:
        for name in input_names:
            paths = sorted(
                p for p in (lambda d: [os.path.join(d, f) for f in os.listdir(d)] if os.path.isdir(d) else [])(os.path.join(calib_dir, name))
                if os.path.basename(p).startswith("sample_")
            )[:max_samples]
            for p in paths:
                stacks[name].append(np.load(p))
        n_samples = min(len(v) for v in stacks.values())
    if n_samples == 0:
        raise FileNotFoundError(f"no calibration samples under {calib_dir}")
    calibration = {name: np.concatenate(arrs, axis=0) for name, arrs in stacks.items() if arrs}
    print(f"calibration samples: {n_samples}")
    for name, arr in calibration.items():
        print(f"  {name}: {arr.shape} {arr.dtype}")
    return calibration


def main():
    global WORK_DIR
    onnx_in, onnx_out, calib_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    max_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    WORK_DIR = tempfile.mkdtemp(prefix="int8wk_")

    _patch_byte_size()
    _patch_save_onnx()
    _patch_infer_shapes()
    _patch_check_model()

    with open(onnx_in + ".metadata.json") as f:
        input_names = json.load(f)["input_names"]
    calibration = load_calibration_dict(calib_dir, input_names, max_samples)

    from modelopt.onnx.quantization import quantize

    # quantize() writes to output_path itself; keep workdir alive during it.
    quantize(
        onnx_path=onnx_in,
        quantize_mode="int8",
        calibration_data=calibration,
        calibration_method="entropy",
        calibration_eps=[os.environ.get("CALIB_EPS", "cpu")],
        use_external_data_format=True,
        keep_io_types=True,
        output_path=onnx_out,
    )

    # The written model's tensors may still be external refs at the ORIGINAL
    # input .data (modelopt passes refs through; onnx.save never writes them).
    # Resolve refs -> raw bytes, then re-export to ONE fresh external file.
    from onnx import external_data_helper as edh

    data_out = onnx_out + ".data"
    if os.path.exists(data_out):
        os.remove(data_out)
    m = onnx.load(onnx_out, load_external_data=False)
    in_dir = os.path.dirname(os.path.abspath(onnx_in))
    # Tensors may point at WORK_DIR-internal names (c.onnx.data / m.onnx.data)
    # after check_model/infer_shapes roundtrips — resolve each ref where its
    # file actually lives (WORK_DIR first, then the input's dir), inline it,
    # then re-export everything to ONE fresh external file.
    for t in m.graph.initializer:
        if t.data_location != onnx.TensorProto.EXTERNAL or t.raw_data:
            continue
        loc = next((e.value for e in t.external_data if e.key == "location"), None)
        if not loc:
            continue
        for base in (WORK_DIR, in_dir, os.path.dirname(os.path.abspath(onnx_out))):
            path = os.path.join(base, loc)
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    f.seek(int(next((e.value for e in t.external_data if e.key == "offset"), 0)))
                    t.raw_data = f.read(int(next((e.value for e in t.external_data if e.key == "length"), 0)))
                t.data_location = onnx.TensorProto.DEFAULT
                del t.external_data[:]
                break
        else:
            raise RuntimeError(f"external data not found for {t.name} ({loc})")
    edh.convert_model_to_external_data(
        m,
        all_tensors_to_one_file=True,
        location=os.path.basename(data_out),
        size_threshold=1024,
    )
    onnx.save(m, onnx_out)
    if os.path.isfile(data_out) and os.path.getsize(data_out) > 0:
        print("INT8_SAVED", onnx_out, os.path.getsize(onnx_out), os.path.getsize(data_out))
    else:
        print("INT8_SAVED_INLINE", onnx_out, os.path.getsize(onnx_out))
    shutil.rmtree(WORK_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
