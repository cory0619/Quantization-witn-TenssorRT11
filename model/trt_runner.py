r"""
model/trt_runner.py : TensorRT 엔진 실행기 (cupy device 버퍼 + pinned host 버퍼 + execute_async_v3)

_4_onnx2trt.py (벤치마크 / 검증), _5_predict.py (폴더 추론), server_inference.py (웹 서비스) 가 같이 쓴다.
- 입력/출력 텐서 이름, dtype, (C,H,W), batch 범위(min/opt/max) 는 엔진의 optimization profile 에서 읽는다 (eval_common 에 의존하지 않음).
- infer(x): x 는 (B,C,H,W) 엔진 입력 dtype (R6 엔진은 uint8 BGR 원본 픽셀). B 는 profile max 이하. 반환: host numpy 사본 (B,1,H,W) uint8.
  pinned host -> device -> 실행 -> device -> pinned host 순서로, 서버 경로와 같다.
- run_all(x, batch): 장수가 많으면 batch 단위로 잘라 infer.
- execute(): 입력이 이미 device 에 있을 때 GPU 실행만 (벤치마크용).
- 실행 컨텍스트와 스트림을 공유하지 않으므로 한 스레드에서만 쓸 것 (서버는 추론 전용 스레드 1개에서 호출).
"""
import gc

import cupy as cp
import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def pinned(shape, dtype):
    """page-locked host 버퍼 (numpy 뷰). host<->device 복사가 pageable 메모리보다 빠르다"""
    n = int(np.prod(shape))
    mem = cp.cuda.alloc_pinned_memory(n * np.dtype(dtype).itemsize)
    return np.frombuffer(mem, dtype=dtype, count=n).reshape(shape)


class TrtRunner:
    def __init__(self, engine_path, logger=TRT_LOGGER):
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to load {engine_path}")
        self.engine_path = engine_path
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.in_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.in_dtype = trt.nptype(self.engine.get_tensor_dtype(self.in_name))
        self.out_dtype = trt.nptype(self.engine.get_tensor_dtype(self.out_name))
        pmin, popt, pmax = self.engine.get_tensor_profile_shape(self.in_name, 0)
        self.batch_min, self.batch_opt, self.batch_max = int(pmin[0]), int(popt[0]), int(pmax[0])
        self.in_chw = tuple(int(v) for v in pmax[1:])            # (C,H,W). H, W 는 고정 profile
        self.profile = f"{self.batch_min}/{self.batch_opt}/{self.batch_max}"
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.cur_batch = None

    @property
    def in_hw(self):
        return self.in_chw[1], self.in_chw[2]

    def set_batch(self, b):
        if self.cur_batch == b:
            return
        if not (self.batch_min <= b <= self.batch_max):
            raise ValueError(f"batch {b} out of engine profile {self.profile}")
        in_shape = (b,) + self.in_chw
        self.ctx.set_input_shape(self.in_name, in_shape)
        out_shape = tuple(self.ctx.get_tensor_shape(self.out_name))      # (b, 1, H, W)
        self.d_in = cp.empty(in_shape, self.in_dtype)
        self.d_out = cp.empty(out_shape, self.out_dtype)
        self.h_in = pinned(in_shape, self.in_dtype)
        self.h_out = pinned(out_shape, self.out_dtype)
        self.ctx.set_tensor_address(self.in_name, self.d_in.data.ptr)
        self.ctx.set_tensor_address(self.out_name, self.d_out.data.ptr)
        self.cur_batch = b
        self.out_shape = out_shape

    def execute(self):
        """GPU 실행만 (입력은 이미 device 에 있음)"""
        self.ctx.execute_async_v3(self.stream.ptr)
        self.stream.synchronize()

    def infer(self, x):
        """pinned host -> device -> 실행 -> device -> pinned host. x: (B,C,H,W) in_dtype, B <= batch_max"""
        if x.shape[1:] != self.in_chw:
            raise ValueError(f"input {x.shape[1:]} != engine {self.in_chw} (C,H,W)")
        b = x.shape[0]
        self.set_batch(b)
        self.h_in[...] = x
        with self.stream:
            self.d_in.set(self.h_in, stream=self.stream)
            self.ctx.execute_async_v3(self.stream.ptr)
            self.d_out.get(stream=self.stream, out=self.h_out)
        self.stream.synchronize()
        return self.h_out.copy()

    def run_all(self, images, batch=1):
        batch = min(batch, self.batch_max)
        return np.concatenate([self.infer(images[i:i + batch]) for i in range(0, len(images), batch)])

    def warmup(self, batch=None):
        b = min(batch or self.batch_opt, self.batch_max)
        self.infer(np.zeros((b,) + self.in_chw, self.in_dtype))

    def activation_mib(self):
        return self.engine.device_memory_size_v2 / 2 ** 20

    def close(self):
        for a in ("ctx", "engine", "d_in", "d_out", "h_in", "h_out"):
            if hasattr(self, a):
                delattr(self, a)
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
