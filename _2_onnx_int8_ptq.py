r"""
_2_onnx_int8_ptq.py : FP32 ONNX -> INT8 (explicit Q/DQ) ONNX, NVIDIA ModelOpt PTQ   (R6 UNet_ConvNeXt)

왜 이 단계가 필요한가:
- TensorRT 11.x 는 calibrator 기반 implicit INT8 (IInt8Calibrator, BuilderFlag.INT8) 이 제거되었다. 빌더(_4) 는 ONNX 에 이미 들어 있는
  QuantizeLinear/DequantizeLinear 노드와 그 scale 상수를 그대로 쓸 뿐, 스스로 "어디를 INT8 로 할지 / scale 이 얼마인지" 를 정하지 않는다.
  _1 의 FP32 ONNX 를 _4 에 바로 넣으면 FP32 엔진이 나온다. 그래서 Q/DQ 를 넣는 도구가 앞단에 있어야 하고, 그것이 이 스크립트의
  ModelOpt (modelopt.onnx.quantization, NVIDIA TensorRT Model Optimizer) 다. "TensorRT 대신 ModelOpt" 가 아니라 분업이다.
- 무엇이 INT8 로 바뀌나: ModelOpt 는 가중치 있는 연산(Conv, MatMul 등) 앞에 Q/DQ 를 넣고 가중치를 per-channel INT8 로 바꾼다.
  ConvNeXt 백본의 stem/다운샘플 Conv, 블록의 7x7 depthwise Conv, MLP 의 Linear 두 개(MatMul), 디코더의 1x1/3x3 Conv 가 모두 대상이다.
  LayerNorm, GELU(Erf), 잔차 덧셈, 입력 정규화, 출력 임계값 등 Q/DQ 가 안 붙는 연산은 high_precision_dtype(fp16) 으로 바뀐다.
  정확히 어느 노드가 INT8 인지는 _3_int8_layer_check.py 가 센다.
- calibration 에는 라벨(json)이 필요 없다. 학습과 같은 전처리(BGR, IMG_SIZE resize) 의 uint8 이미지를 넣으면 된다. test 폴더는 평가용이므로 넣지 말 것.
  ModelOpt 는 ORT 를 패치해 이미지 1장씩 중간 텐서를 받아 히스토그램에 누적하므로, 전체 해상도 그래프라도 호스트 메모리는 1장분(수 GB)만 쓴다.
- 변환 후 FP32 ONNX 와 INT8 ONNX 를 ONNX Runtime 으로 test set 에서 비교: 클래스별 정확도, 마스크 일치율, 추론 속도, GPU 메모리/사용률 -> 콘솔 (정확도와 환경은 리포트 json/html 에도)
  * ORT 의 INT8 Q/DQ 실행은 "시뮬레이션"(DQ 후 FP16 연산) 이라 속도는 배포 속도가 아니다. 실제 INT8 속도는 _4 의 TensorRT 엔진 결과를 볼 것.

파이프라인:  _1_model2onnx.py  ->  _2_onnx_int8_ptq.py  ->  _3_int8_layer_check.py  ->  _4_onnx2trt.py

사용:
    python _2_onnx_int8_ptq.py                       # 양자화 + 비교
    python _2_onnx_int8_ptq.py --compare-only        # 이미 만든 INT8 ONNX 로 비교만
    python _2_onnx_int8_ptq.py --calib-method max    # entropy 대신 max (MinMax symmetric) calibration
    python _2_onnx_int8_ptq.py --calib-eps cpu       # calibration 을 CPU 에서 (GPU 메모리 부족 시; 느림)
    python _2_onnx_int8_ptq.py --exclude /outc/Conv  # 특정 노드는 양자화 제외 (정확도 저하 시 시도)
"""
import cudnn_fix  # torch / onnxruntime 보다 먼저: calibration 과 검증이 CUDA 에서 돌게 함 (cudnn_fix.py 참고)
import argparse
import gc
import os
import platform
import time

import numpy as np
import onnxruntime as ort

from eval_common import (INPUT_NAME, DATASET_DIR, TEST_DIR, CALIB_DIRS, ONNX_FP32, ONNX_INT8, THRESH, CLASSES, IMG_SIZE,
                         load_images, load_test_set, unpack_masks, metrics, agreement, iou_summary,
                         print_accuracy_table, benchmark, GpuMonitor, fmt_mib, fmt_bytes, save_report)

# ======================================================
# 1. 설정
# ======================================================
MAX_CALIB      = 64                # calibration 이미지 상한 (dataset_pcb/train 은 22장이라 전부 사용)
CALIB_METHOD   = "entropy"         # 예전 IInt8EntropyCalibrator2 에 해당. "max" 도 가능 (--calib-method)
CALIB_EPS      = ["cuda:0", "cpu"] # calibration 실행 provider 우선순위 (--calib-eps cpu 로 CPU 강제)
HIGH_PRECISION = "fp16"            # 양자화되지 않는 나머지 연산의 정밀도 (예전 FP16 fallback 플래그 역할)

RUN_BATCH      = 1                 # 정확도 계산 시 ORT 배치 (전체 해상도라 1)
BENCH_BATCHES  = [1, 4, 8]         # 속도 측정 batch (_4 의 엔진 profile 1/4/8 과 동일). GPU 메모리 부족 시 해당 batch 는 건너뜀
BENCH_WARMUP, BENCH_ITERS = 5, 20
N_CALIB = None                     # quantize() 가 실제로 쓴 calibration 장수 (리포트용)


# ======================================================
# 2. ModelOpt INT8 PTQ (Q/DQ 삽입 + calibration + 나머지 FP16 변환)
# ======================================================
def quantize(args):
    import modelopt.onnx.quantization as moq
    global N_CALIB
    calib, calib_paths = load_images(CALIB_DIRS, args.max_calib)
    N_CALIB = len(calib_paths)
    print(f"[calib] {calib.shape[0]} images from {CALIB_DIRS}, shape {calib.shape} {calib.dtype} (BGR 원본 픽셀, /255·정규화는 그래프 안), "
          f"method {args.calib_method}, eps {args.calib_eps}, 나머지 {HIGH_PRECISION}"
          + (f", exclude {args.exclude}" if args.exclude else ""))
    kwargs = dict(
        onnx_path=ONNX_FP32,
        quantize_mode="int8",
        calibration_data={INPUT_NAME: calib},    # 키 = ONNX 입력 이름. ModelOpt 가 입력 shape 의 batch(=1) 단위로 잘라 넣음
        calibration_method=args.calib_method,
        calibration_eps=list(args.calib_eps),
        high_precision_dtype=HIGH_PRECISION,
        output_path=ONNX_INT8,
    )
    if args.exclude:
        kwargs["nodes_to_exclude"] = list(args.exclude)
    t0 = time.perf_counter()
    try:
        moq.quantize(**kwargs)
    except Exception as e:
        if args.calib_eps[0] != "cpu":
            print(f"[warn] calibration on {args.calib_eps} failed: {type(e).__name__}: {str(e)[:300]}\n[warn] -> CPU 에서 재시도")
            kwargs["calibration_eps"] = ["cpu"]
            moq.quantize(**kwargs)
        else:
            raise
    print(f"[OK] INT8 Q/DQ ONNX: {ONNX_INT8} ({fmt_bytes(ONNX_INT8)}, FP32 는 {fmt_bytes(ONNX_FP32)})  ({time.perf_counter() - t0:.0f}s)")


# ======================================================
# 3. 비교: FP32 ONNX vs INT8 ONNX (ONNX Runtime CUDA EP)
# ======================================================
def make_session(path):
    sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    return sess, sess.get_providers()[0].replace("ExecutionProvider", "")


def run_session(sess, images, batch=RUN_BATCH):
    outs = [sess.run(None, {INPUT_NAME: images[i:i + batch]})[0] for i in range(0, len(images), batch)]
    return np.concatenate(outs)                       # uint8 (N,1,H,W) 비트 패킹


def compare(args):
    images, gts, _ = load_test_set(TEST_DIR)
    gpu = GpuMonitor()
    base_mib = GpuMonitor.process_mib()               # 세션 생성 전 (CUDA context 없음)

    acc_rows, perf_rows, preds = [], [], {}
    for name, path in (("FP32 ONNX", ONNX_FP32), ("INT8 ONNX", ONNX_INT8)):
        sess, provider = make_session(path)

        # --- 정확도 (출력이 이미 임계값 적용된 마스크)
        t0 = time.perf_counter()
        preds[name] = run_session(sess, images)
        print(f"[run] {name:10s} {provider:5s} test {len(images)} 장: {(time.perf_counter() - t0) / len(images) * 1e3:.0f} ms/img (첫 실행 포함)")
        m = metrics(unpack_masks(preds[name]), gts)
        acc_rows.append((name, provider, m))
        print(f"[acc] {name:10s} {iou_summary(m)}")

        # --- 속도 / GPU (sess.run: host->device 복사 + 추론 + device->host 복사 포함)
        for b in args.bench_batches:
            x = np.ascontiguousarray(np.resize(images, (b,) + images.shape[1:]))
            feed = {INPUT_NAME: x}
            try:
                sess.run(None, feed)                                  # 메모리 할당 (OOM 이면 여기서 예외)
                with gpu.sample_util() as util:
                    st = benchmark(lambda: sess.run(None, feed), BENCH_WARMUP, BENCH_ITERS)
            except Exception as e:
                print(f"[bench] {name:10s} {provider:5s} batch {b:4d}: 실패 ({type(e).__name__}: {str(e)[:120]}) -> 건너뜀")
                perf_rows.append((name, provider, b, None, float("nan"), float("nan"), GpuMonitor.process_mib()))
                continue
            perf_rows.append((name, provider, b, st, util.mean, util.max, GpuMonitor.process_mib()))
            print(f"[bench] {name:10s} {provider:5s} batch {b:4d}: {st['mean']:9.2f} ms/batch "
                  f"({st['mean'] / b:8.2f} ms/img, {b / st['mean'] * 1e3:6.1f} img/s) "
                  f"util {util.mean:4.0f}% | proc GPU mem {fmt_mib(perf_rows[-1][-1])}")
        del sess
        gc.collect()

    agree = agreement(preds["FP32 ONNX"], preds["INT8 ONNX"])
    print()
    print_accuracy_table(acc_rows, len(images))
    print("pixel agreement FP32 vs INT8 (ORT): " + ", ".join(f"{k} {v * 100:.2f}%" for k, v in agree.items()))

    # --- 리포트 (json -> html): 환경 + ORT 정확도. ORT 속도는 Q/DQ 시뮬레이션이라 리포트에 싣지 않음 (콘솔만)
    env = gpu.info()
    n_calib = N_CALIB if N_CALIB is not None else len(load_images(CALIB_DIRS, args.max_calib)[1])
    save_report("env", {
        "model": "R6 UNet_ConvNeXt (ConvNeXt-Base 백본 + 덧셈 skip 디코더)", "dataset": os.path.basename(os.path.normpath(DATASET_DIR)),
        "test_dir": TEST_DIR, "n_test": len(images), "img_size": list(IMG_SIZE), "classes": list(CLASSES), "thresh": THRESH,
        "gpu": env.get("gpu"), "driver": env.get("driver"), "gpu_total_mib": env.get("total_mib"),
        "os": platform.platform(), "python": platform.python_version(), "onnxruntime": ort.__version__,
        "modelopt": _modelopt_version(), "calib_method": args.calib_method, "n_calib": n_calib, "calib_dirs": list(CALIB_DIRS),
        "high_precision": HIGH_PRECISION, "exclude": args.exclude,
        "onnx_fp32": ONNX_FP32, "onnx_fp32_mb": os.path.getsize(ONNX_FP32) / 2**20,
        "onnx_int8": ONNX_INT8, "onnx_int8_mb": os.path.getsize(ONNX_INT8) / 2**20,
    })
    save_report("ort", {
        "provider": acc_rows[0][1], "batch": RUN_BATCH,
        "accuracy": {"fp32": acc_rows[0][2], "int8": acc_rows[1][2]},
        "agreement": agree,
        "speed": [{"model": n, "provider": p, "batch": b, "ms_batch": (st["mean"] if st else None), "util_mean": um, "util_max": ux, "proc_mib": mm}
                  for n, p, b, st, um, ux, mm in perf_rows],
    })

def _modelopt_version():
    try:
        import modelopt
        return modelopt.__version__
    except Exception:
        return "?"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ModelOpt INT8 PTQ + FP32/INT8 ONNX 비교 (R6 UNet_ConvNeXt)")
    ap.add_argument("--compare-only", action="store_true", help="양자화는 건너뛰고 기존 INT8 ONNX 로 비교만")
    ap.add_argument("--calib-method", default=CALIB_METHOD, choices=["entropy", "max"], help=f"calibration 방법 (기본 {CALIB_METHOD})")
    ap.add_argument("--calib-eps", nargs="+", default=CALIB_EPS, help=f"calibration 실행 provider (기본 {CALIB_EPS})")
    ap.add_argument("--max-calib", type=int, default=MAX_CALIB, help=f"calibration 이미지 상한 (기본 {MAX_CALIB})")
    ap.add_argument("--exclude", nargs="*", default=None, help="양자화에서 제외할 노드 이름 (ModelOpt nodes_to_exclude, 패턴 가능)")
    ap.add_argument("--bench-batches", type=int, nargs="+", default=BENCH_BATCHES, help=f"속도 측정 batch (기본 {BENCH_BATCHES})")
    args = ap.parse_args()
    if not args.compare_only:
        quantize(args)
    elif not os.path.exists(ONNX_INT8):
        raise SystemExit(f"{ONNX_INT8} 가 없음. --compare-only 없이 실행할 것")
    compare(args)
