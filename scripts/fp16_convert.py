#!/usr/bin/env python3
"""FP16 AutoCast conversion for RTPolicy policy ONNX (cluster-side).

All modelopt large-model paths (infer_shapes / check_model / save_onnx) are
patched to roundtrip through ONE persistent temp dir so external weights stay
on disk throughout; the final save re-points tensors at a fresh external file
next to the output.

Usage: python3 fp16_convert.py <in.onnx> <out.onnx>
"""
import os
import shutil
import sys
import tempfile

import onnx
from google.protobuf.message import EncodeError
from modelopt.onnx import utils as onnx_utils

WORK_DIR = None  # single persistent dir for every temp roundtrip


def _patch_byte_size():
    def byte_size_patched(self, **kwargs):
        total = 0
        for tensor in self.graph.initializer:
            total += len(tensor.raw_data) if tensor.raw_data else 8
        return max(total * 2, 3 * (1024**3))  # force every large-model branch
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
            data_path = os.path.join(WORK_DIR, os.path.basename(onnx_path) + ".data")
            if os.path.exists(data_path):
                os.remove(data_path)
            onnx.save_model(
                model,
                os.path.join(WORK_DIR, os.path.basename(onnx_path)),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=os.path.basename(data_path),
                size_threshold=1024,
            )
            # Move the saved pair to the requested path.
            shutil.move(os.path.join(WORK_DIR, os.path.basename(onnx_path)), onnx_path)
            # External data stays in WORK_DIR under its own name.
        else:
            onnx.save(model, onnx_path)
    onnx_utils.save_onnx = save_onnx_patched


def _patch_infer_shapes():
    def infer_shapes_patched(model, **kwargs):
        orig_path = os.path.join(WORK_DIR, "m.onnx")
        inferred_path = os.path.join(WORK_DIR, "i.onnx")
        onnx_utils.save_onnx(model, orig_path, save_as_external_data=True)
        onnx.shape_inference.infer_shapes_path(orig_path, inferred_path, **kwargs)
        # Point tensors at m.onnx.data (the only live data file) and inline-load.
        return onnx.load(inferred_path, load_external_data=True)
    onnx_utils.infer_shapes = infer_shapes_patched


def _patch_check_model():
    def check_model_patched(model):
        path = os.path.join(WORK_DIR, "c.onnx")
        onnx_utils.save_onnx(model, path, save_as_external_data=True)
        onnx.checker.check_model(path)
        return onnx.load(path, load_external_data=True)
    onnx_utils.check_model = check_model_patched


def main():
    global WORK_DIR
    onnx_in, onnx_out = sys.argv[1], sys.argv[2]
    WORK_DIR = tempfile.mkdtemp(prefix="fp16wk_")

    _patch_byte_size()
    _patch_save_onnx()
    _patch_infer_shapes()
    _patch_check_model()

    from modelopt.onnx.autocast import convert as autocast_convert

    model = autocast_convert.convert_to_mixed_precision(
        onnx_path=onnx_in,
        low_precision_type="fp16",
        keep_io_types=True,
        data_max=float("inf"),
        op_types_to_exclude=[],
        providers=["cpu"],
    )

    # model's tensors may still be external refs at the ORIGINAL input .data
    # (modelopt passes refs through; onnx.save_model never writes data for them).
    # Resolve refs -> raw bytes from the input's dir, then re-export to ONE new
    # external file next to onnx_out.
    from onnx import external_data_helper as edh

    out_data = onnx_out + ".data"
    if os.path.exists(out_data):
        os.remove(out_data)
    in_dir = os.path.dirname(os.path.abspath(onnx_in))
    # Tensors may point at WORK_DIR-internal names (c.onnx.data / m.onnx.data)
    # after check_model/infer_shapes roundtrips — resolve each ref where its
    # file actually lives (WORK_DIR first, then the input's dir), inline it,
    # then re-export everything to ONE fresh external file.
    for t in model.graph.initializer:
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
        model,
        all_tensors_to_one_file=True,
        location=os.path.basename(out_data),
        size_threshold=1024,
    )
    onnx.save(model, onnx_out)
    if not os.path.isfile(out_data) or os.path.getsize(out_data) == 0:
        raise RuntimeError("final save did not materialize data")

    print("FP16_SAVED", onnx_out, os.path.getsize(onnx_out), os.path.getsize(out_data))
    shutil.rmtree(WORK_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
