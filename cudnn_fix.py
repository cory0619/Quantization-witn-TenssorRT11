r"""
cudnn_fix.py  -  torch / onnxruntime 를 import 하기 "전에" import 할 것.   (2026-10-15)

문제: torch 휠은 torch/lib 에 cuDNN 9.20 을 번들하는데, Windows 사용자 PATH 에 시스템 cuDNN 9.27 폴더
      (C:\Program Files\NVIDIA\CUDNN\v9.27\bin\12.9\x64) 가 있으면 서브라이브러리 하나
      (cudnn_engines_tensor_ir64_9.dll) 가 9.27 에서 로드되어 모든 CUDA conv 가
      CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH 로 실패한다. 같은 이유로 torch 를 import 한 프로세스에서는
      onnxruntime-gpu 의 CUDA provider 도 조용히 CPU 로 떨어진다.

해결 (이 프로세스 안에서만, Windows PATH 자체는 건드리지 않음):
  1. 프로세스 PATH 에서 "...\CUDNN\v9..." 항목을 제거한다 -> torch 는 번들 9.20 만 쓴다.
  2. torch/lib 의 cuDNN DLL 들을 전체 경로로 미리 로드한다 -> torch 없이 onnxruntime-gpu 만 쓰는
     프로세스도 같은 cuDNN 을 찾는다. (나중에 import torch 가 같은 파일을 다시 로드해도 재사용된다.)

사용:  스크립트 맨 위에서  import cudnn_fix
"""
import ctypes
import glob
import os
import re
import sys

_SYSTEM_CUDNN = re.compile(r"[\\/]cudnn[\\/]v9", re.IGNORECASE)


def _strip_system_cudnn_from_path():
    sep = os.pathsep
    parts = os.environ.get("PATH", "").split(sep)
    kept = [p for p in parts if not _SYSTEM_CUDNN.search(p)]
    if len(kept) != len(parts):
        os.environ["PATH"] = sep.join(kept)


def _preload_torch_cudnn():
    for sp in sys.path:
        torch_lib = os.path.join(sp, "torch", "lib")
        if not os.path.isdir(torch_lib):
            continue
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(torch_lib)
        pending = sorted(glob.glob(os.path.join(torch_lib, "cudnn*.dll")))
        for _ in range(4):  # 의존 순서를 모르므로 실패한 것은 몇 번 재시도
            failed = []
            for dll in pending:
                try:
                    ctypes.WinDLL(dll)
                except OSError:
                    failed.append(dll)
            pending = failed
            if not pending:
                break
        return


def apply():
    try:
        _strip_system_cudnn_from_path()
    except Exception:
        pass
    try:
        _preload_torch_cudnn()
    except Exception:
        pass


apply()
