r"""
_4_onnx2trt.py : ONNX -> TensorRT 11.x 엔진 빌드 + 검증 + FP32/INT8 엔진 속도·GPU 비교   (R6 UNet_ConvNeXt)

- INT8: _2_onnx_int8_ptq.py 가 만든 Q/DQ ONNX 를 그대로 파싱한다. TRT 11 은 strongly typed 네트워크가 기본이라
  정밀도는 ONNX 텐서 dtype / Q-DQ 노드가 결정한다. 예전처럼 BuilderFlag.INT8/FP16, calibrator, layer.precision 을
  지정하는 코드는 없다(API 자체가 제거됨). 빌더가 하는 일은 parse + optimization profile + 빌드뿐이다.
  -> 그래서 _1 의 FP32 ONNX 를 여기 넣으면 FP32 엔진, _2 의 Q/DQ ONNX 를 넣어야 INT8 엔진이 된다.
- FP32: 비교용으로 _1 의 FP32 ONNX 로도 같은 profile 로 엔진을 만든다 (--skip-fp32 로 생략).
- 입출력: uint8 (B,3,1088,1344) BGR 원본 픽셀 -> uint8 (B,1,1088,1344) 비트 패킹 마스크 (bit0 pcb, bit1 terminal, bit2 hole).
  서버는 float 가 아니라 이 uint8 버퍼를 그대로 넣고 받으면 된다 (복사량 FP32 대비 1/4, 출력은 1 byte/px).
- 검증: 엔진 출력을 같은 ONNX 의 ONNX Runtime 출력과 비교(클래스별 마스크 일치율) + test set 클래스별 정확도.
- 속도/GPU: batch 별 엔진 실행 시간(GPU only / 복사 포함), GPU 사용률, 프로세스 GPU 메모리, 엔진 activation 메모리.
  결과는 콘솔과 리포트 json/html (eval_common.save_report -> report_html.py: 속도 / GPU 메모리 / 정확도) 에 기록한다.
- 빌드 시간 단축: timing cache (TIMING_CACHE) 를 저장/재사용.

파이프라인:  _1_model2onnx.py  ->  _2_onnx_int8_ptq.py  ->  _3_int8_layer_check.py  ->  _4_onnx2trt.py

사용:
    python _4_onnx2trt.py                      # INT8 + FP32(비교용) 엔진 빌드 (profile batch 1/4/8), 검증, 비교
    python _4_onnx2trt.py --skip-fp32          # INT8 만
    python _4_onnx2trt.py --skip-build         # 이미 있는 엔진으로 검증/비교만
    python _4_onnx2trt.py --batch-opt 4 --batch-max 8     # profile 변경 (서버 batch 가 max 이하여야 함)
    python _3_int8_layer_check.py --engine     # 엔진 레이어 정밀도 (DETAILED 로 빌드되어 있음)
    python server_inference.py                 # 만든 엔진으로 웹 서비스 (공유메모리 + Flask, batch 추론)
    python _5_predict.py                       # dense 체크포인트 + 만든 엔진으로 폴더 추론 (이미지별 지표 + 오버레이, 비교)
"""
import cudnn_fix  # onnxruntime(검증용) 보다 먼저 (cudnn_fix.py 참고). TensorRT 자체는 cuDNN 을 쓰지 않음
import argparse
import gc
import os
import time

import cupy as cp
import numpy as np
import onnxruntime as ort
import tensorrt as trt

from eval_common import (INPUT_NAME, TEST_DIR, HEIGHT, WIDTH, IN_CHANNELS, CLASSES, THRESH,
                         ONNX_FP32, ONNX_INT8, ENGINE_FP32, ENGINE_INT8, TIMING_CACHE,
                         load_test_set, unpack_masks, metrics, agreement, iou_summary,
                         print_accuracy_table, benchmark, GpuMonitor, fmt_mib, fmt_bytes, save_report)
from model.trt_runner import TrtRunner

# ======================================================
# 1. 설정
# ======================================================
# dynamic batch optimization profile (min / opt / max), INT8 / FP32 엔진 공통. (--batch-opt / --batch-max 로 변경)
# * 전체 해상도 ConvNeXt-Base 라 R5 때의 1/128/256 같은 큰 batch 는 의미가 없다. 서버 batch 는 max 이하여야 한다.
BATCH_MIN, BATCH_OPT, BATCH_MAX = 1, 4, 8
WORKSPACE_GB = 6                  # 빌드 중 tactic 탐색 scratch 상한 (런타임 메모리 아님)

BENCH_BATCHES = [1, 4, 8]         # 속도 측정 batch. 엔진의 max batch 보다 큰 값은 제외
BENCH_WARMUP, BENCH_ITERS = 10, 50
ORT_BATCH = 1                     # 검증용 ONNX Runtime 배치 (전체 해상도라 1)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ======================================================
# 2. 엔진 빌드 (parse + profile 만. 정밀도는 ONNX 가 결정)
# ======================================================
def build_engine(onnx_path, engine_path, batch_max, batch_opt=None):
    batch_opt = batch_opt or batch_max
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print("PARSE ERROR:", parser.get_error(i))
        raise RuntimeError(f"ONNX parse failed: {onnx_path}")
    print(f"[parse] {onnx_path}: {network.num_layers} layers, input {network.get_input(0).name} {network.get_input(0).dtype} {network.get_input(0).shape}, "
          f"output {network.get_output(0).name} {network.get_output(0).dtype} {network.get_output(0).shape}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_GB << 30)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED   # _3 --engine 으로 레이어 정밀도 확인용
    profile = builder.create_optimization_profile()
    shp = lambda b: (b, IN_CHANNELS, HEIGHT, WIDTH)
    profile.set_shape(INPUT_NAME, shp(BATCH_MIN), shp(batch_opt), shp(batch_max))
    config.add_optimization_profile(profile)

    # timing cache: 같은 레이어 구성의 tactic 측정 결과를 재사용 (FP32/INT8 두 엔진, 재빌드 모두 혜택)
    cache_bytes = b""
    if os.path.exists(TIMING_CACHE):
        with open(TIMING_CACHE, "rb") as f:
            cache_bytes = f.read()
    cache = config.create_timing_cache(cache_bytes)
    config.set_timing_cache(cache, ignore_mismatch=False)

    print(f"[build] {onnx_path} -> {engine_path}  (profile batch {BATCH_MIN}/{batch_opt}/{batch_max}, strongly typed, "
          f"workspace {WORKSPACE_GB} GB, timing cache {'재사용 ' + str(len(cache_bytes) // 1024) + ' KB' if cache_bytes else '새로 생성'})")
    t0 = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"engine build failed: {onnx_path} (batch max {batch_max})")
    with open(engine_path, "wb") as f:
        f.write(plan)
    try:
        with open(TIMING_CACHE, "wb") as f:
            f.write(bytes(memoryview(config.get_timing_cache().serialize())))
    except Exception as e:
        print(f"[warn] timing cache 저장 실패: {e}")
    sec = time.perf_counter() - t0
    print(f"[OK] engine {engine_path}  {plan.nbytes / 2**20:.1f} MB, build {sec:.0f}s, TensorRT {trt.__version__}")
    return sec


# ======================================================
# 3. 엔진 실행: model/trt_runner.py 의 TrtRunner (_5, server_inference.py 와 공유)
# ======================================================
def ort_masks(onnx_path, images, batch=ORT_BATCH):
    sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    prov = sess.get_providers()[0].replace("ExecutionProvider", "")
    out = np.concatenate([sess.run(None, {INPUT_NAME: images[i:i + batch]})[0] for i in range(0, len(images), batch)])
    del sess
    gc.collect()
    return out, prov


# ======================================================
# 4. main
# ======================================================
def main():
    ap = argparse.ArgumentParser(description="ONNX -> TensorRT 엔진 빌드 + 검증 + FP32/INT8 비교 (R6 UNet_ConvNeXt)")
    ap.add_argument("--skip-build", action="store_true", help="기존 엔진 파일로 검증/비교만")
    ap.add_argument("--skip-fp32", action="store_true", help="FP32 비교 엔진 생략")
    ap.add_argument("--batch-opt", type=int, default=BATCH_OPT, help=f"profile opt batch (기본 {BATCH_OPT})")
    ap.add_argument("--batch-max", type=int, default=BATCH_MAX, help=f"profile max batch (기본 {BATCH_MAX})")
    ap.add_argument("--fp32-batch-max", type=int, default=None, help="FP32 비교 엔진만 max batch 를 다르게 (기본: --batch-max 와 동일)")
    ap.add_argument("--bench-batches", type=int, nargs="+", default=BENCH_BATCHES, help=f"속도 측정 batch (기본 {BENCH_BATCHES})")
    args = ap.parse_args()

    # name, onnx, engine, batch_max
    targets = [("INT8 engine", ONNX_INT8, ENGINE_INT8, args.batch_max)]
    if not args.skip_fp32:
        targets.insert(0, ("FP32 engine", ONNX_FP32, ENGINE_FP32, args.fp32_batch_max or args.batch_max))

    build_sec = {}
    for name, onnx_path, engine_path, bmax in targets:
        if args.skip_build and os.path.exists(engine_path):
            print(f"[skip build] {engine_path}")
            continue
        if not os.path.exists(onnx_path):
            raise SystemExit(f"{onnx_path} 가 없음 ({'_1' if onnx_path == ONNX_FP32 else '_2'} 를 먼저 실행)")
        build_sec[name] = build_engine(onnx_path, engine_path, bmax, min(args.batch_opt, bmax))

    # ---------- 엔진 실행: 정확도용 출력 + 속도 + GPU 메모리 ----------
    images, gts, _ = load_test_set(TEST_DIR)
    gpu = GpuMonitor()
    cp.cuda.Device(0).synchronize()                 # CUDA context 생성
    base_mib = GpuMonitor.process_mib()
    print(f"[gpu] process GPU memory after CUDA context: {fmt_mib(base_mib)}")

    trt_out, perf_rows, eng_rows = {}, [], []
    for name, onnx_path, engine_path, _ in targets:
        r = TrtRunner(engine_path)
        t0 = time.perf_counter()
        trt_out[name] = r.run_all(images)
        print(f"[run] {name:12s} test {len(images)} 장 batch 1: {(time.perf_counter() - t0) / len(images) * 1e3:.1f} ms/img (첫 실행 포함), "
              f"output {trt_out[name].dtype} {trt_out[name].shape}")
        load_mib = GpuMonitor.process_mib()
        peak = load_mib
        for b in [b for b in args.bench_batches if b <= r.batch_max]:
            x = np.ascontiguousarray(np.resize(images, (b,) + images.shape[1:]))
            r.infer(x)                              # 버퍼 할당 + shape 설정
            with gpu.sample_util() as util_g:
                st_gpu = benchmark(r.execute, BENCH_WARMUP, BENCH_ITERS)
            with gpu.sample_util() as util_e:
                st_e2e = benchmark(lambda: r.infer(x), BENCH_WARMUP, BENCH_ITERS)
            mib = GpuMonitor.process_mib()
            peak = max(peak or 0, mib or 0)
            perf_rows.append((name, b, st_gpu, st_e2e, util_g.mean, util_g.max, mib))
            print(f"[bench] {name:12s} batch {b:3d}: GPU {st_gpu['mean']:8.2f} ms | e2e {st_e2e['mean']:8.2f} ms/batch "
                  f"({st_e2e['mean'] / b:7.2f} ms/img, {b / st_e2e['mean'] * 1e3:6.1f} img/s) "
                  f"util {util_g.mean:4.0f}% | proc GPU mem {fmt_mib(mib)}")
        eng_rows.append((name, engine_path, r.profile, os.path.getsize(engine_path) / 2**20,
                         r.activation_mib(), r.in_dtype.__name__, r.out_dtype.__name__, load_mib, peak, build_sec.get(name)))
        print(f"[engine] {name:12s} profile {r.profile}, file {os.path.getsize(engine_path) / 2**20:.1f} MB, "
              f"activation memory {r.activation_mib():.0f} MiB (max batch), I/O {r.in_dtype.__name__}/{r.out_dtype.__name__}, "
              f"proc GPU mem load {fmt_mib(load_mib)} / peak {fmt_mib(peak)}")
        r.close()

    # ---------- 검증: 엔진 vs 같은 ONNX 의 ONNX Runtime 출력, test set 정확도 ----------
    acc_rows, val_rows = [], []
    for name, onnx_path, _, _ in targets:
        ref, prov = ort_masks(onnx_path, images)
        ag = agreement(trt_out[name], ref)
        val_rows.append((name, prov, ag))
        m = metrics(unpack_masks(trt_out[name]), gts)
        acc_rows.append((name, "TensorRT", m))
        print(f"[validate] {name:12s} vs ONNX Runtime({prov}) 마스크 일치율: " + ", ".join(f"{k} {v * 100:.3f}%" for k, v in ag.items()))
        print(f"[acc] {name:12s} {iou_summary(m)}")
    agree_32_8 = None
    if len(targets) == 2:
        agree_32_8 = agreement(trt_out[targets[0][0]], trt_out[targets[1][0]])
        print("[validate] FP32 engine vs INT8 engine 마스크 일치율: " + ", ".join(f"{k} {v * 100:.3f}%" for k, v in agree_32_8.items()))
    print()
    print_accuracy_table(acc_rows, len(images))

    # ---------- 리포트 (json -> html): 속도, GPU 메모리, 정확도 ----------
    env = gpu.info()
    key = {n: ("fp32" if n.startswith("FP32") else "int8") for n, *_ in targets}
    save_report("trt", {
        "trt": trt.__version__, "gpu": env.get("gpu"), "driver": env.get("driver"), "gpu_total_mib": env.get("total_mib"),
        "workspace_gb": WORKSPACE_GB, "cupy": cp.__version__, "n_test": len(images), "base_mib": base_mib,
        "io": f"uint8 (B,{IN_CHANNELS},{HEIGHT},{WIDTH}) BGR -> uint8 (B,1,{HEIGHT},{WIDTH}) 비트 패킹 "
              f"({', '.join(f'bit{i} {c}' for i, c in enumerate(CLASSES))}), sigmoid > {THRESH} 는 엔진 안",
        "bench": {"batches": list(args.bench_batches), "warmup": BENCH_WARMUP, "iters": BENCH_ITERS},
        "engines": {key[n]: {"name": n, "file": os.path.basename(p), "profile": prof, "size_mb": sz, "activation_mib": act,
                             "in_dtype": i, "out_dtype": o, "load_mib": l, "peak_mib": pk, "build_s": bs}
                    for n, p, prof, sz, act, i, o, l, pk, bs in eng_rows},
        "speed": [{"engine": key[n], "batch": b, "gpu_ms": g["mean"], "gpu_ms_median": g["median"], "gpu_ms_std": g["std"],
                   "e2e_ms": e["mean"], "util_mean": um, "util_max": ux, "proc_mib": mm}
                  for n, b, g, e, um, ux, mm in perf_rows],
        "accuracy": {key[n]: m for n, _, m in acc_rows},
        "validation": {key[n]: {"provider": p, "agreement": ag} for n, p, ag in val_rows},
        "agreement_fp32_int8": agree_32_8,
    })

if __name__ == "__main__":
    main()
