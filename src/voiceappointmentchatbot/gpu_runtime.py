"""Make NVIDIA CUDA DLLs visible to CTranslate2 on Windows.

The ``nvidia-cublas-cu12``, ``nvidia-cudnn-cu12`` and ``nvidia-cuda-nvrtc-cu12``
wheels ship the runtime DLLs inside ``site-packages/nvidia/<lib>/bin/``.
Python on Windows does not add those directories to the DLL search path
automatically, so CTranslate2 fails to load ``cublas64_12.dll`` and
similar libraries unless we register them up front. This module is
imported eagerly by :mod:`voiceappointmentchatbot.config` so the search
path is patched before any CUDA-using import (faster-whisper, ctranslate2)
is resolved.

On non-Windows platforms or when the wheels are not installed (CPU-only
deployments) the module is a no-op.
"""

from typing import List
import os
import sys
import sysconfig


def _candidate_directories() -> List[str]:
    """Return absolute paths to NVIDIA wheel ``bin`` directories that exist.

    Returns:
        Each path is guaranteed to exist on disk. The list is empty when
        no NVIDIA runtime wheels are installed in the active environment.
    """
    site_packages = sysconfig.get_paths()["purelib"]
    nvidia_root = os.path.join(site_packages, "nvidia")
    if not os.path.isdir(nvidia_root):
        return []

    directories: List[str] = []
    for entry in os.listdir(nvidia_root):
        bin_dir = os.path.join(nvidia_root, entry, "bin")
        if os.path.isdir(bin_dir):
            directories.append(bin_dir)
    return directories


def register_cuda_dlls() -> List[str]:
    """Add bundled NVIDIA DLL directories to the Windows DLL search path.

    Patches both ``os.add_dll_directory`` (for direct loads) and the
    ``PATH`` environment variable (for transitive ``LoadLibraryW`` calls
    issued by CTranslate2 once it has been loaded), because the former
    alone does not cover dependencies resolved at runtime by another DLL.

    Returns:
        The list of directories that were registered. Empty on non-Windows
        platforms or when no NVIDIA runtime wheels are present.
    """
    if sys.platform != "win32":
        return []

    directories = _candidate_directories()
    if not directories:
        return []

    for directory in directories:
        try:
            os.add_dll_directory(directory)
        except (OSError, FileNotFoundError):
            pass

    existing = os.environ.get("PATH", "")
    prefix = os.pathsep.join(directories)
    if prefix not in existing:
        os.environ["PATH"] = prefix + os.pathsep + existing

    return directories


_REGISTERED_DIRS: List[str] = register_cuda_dlls()
