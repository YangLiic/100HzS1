#!/usr/bin/env bash
# Source before importing SAPIEN. All paths are independent of other checkouts.
export SAPIEN_EGL_NO_X11=1 MUJOCO_GL=egl
export KRAKEN_EGL_DEVICE="${GPU_ID:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
if [ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ] && [ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
fi
if [ -z "${VK_ICD_FILENAMES:-}" ]; then
  # Headless cloud images may ship an unusable GLX ICD. Prefer NVIDIA EGL.
  for library in /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 /usr/lib64/libEGL_nvidia.so.0; do
    if [ -e "$library" ]; then
      icd_dir="${TMPDIR:-/tmp}/vfa_robotwin_icd_${UID}"
      mkdir -p "$icd_dir"
      printf '{"file_format_version":"1.0.0","ICD":{"library_path":"%s","api_version":"1.3.242"}}\n' "$library" > "$icd_dir/nvidia.json"
      export VK_ICD_FILENAMES="$icd_dir/nvidia.json"
      break
    fi
  done
  if [ -z "${VK_ICD_FILENAMES:-}" ]; then
    for candidate in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do
      if [ -f "$candidate" ]; then export VK_ICD_FILENAMES="$candidate"; break; fi
    done
  fi
fi

export CUROBO_TORCH_COMPILE_DISABLE=1
