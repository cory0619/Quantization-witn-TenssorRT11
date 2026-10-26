r"""
_1_model2onnx.py : FP32 체크포인트(.pt) -> 배포용 FP32 ONNX   (R6 UNet_ConvNeXt, deploy 그래프)

- deploy=True 로 export 한다. 입력 uint8 (B,3,1088,1344) BGR 원본 픽셀 (/255 와 ImageNet 정규화는 그래프 안),
  출력 uint8 (B,1,1088,1344): 클래스별 sigmoid > THRESH 마스크를 비트로 패킹 (bit0 pcb, bit1 terminal, bit2 hole).
  eval_common.unpack_masks 로 (B,3,H,W) 복원. 서버는 float 입출력이 아니라 이 uint8 인터페이스를 써야 한다.
- H, W 는 고정 (TensorRT profile 도 고정), batch 만 dynamic.
- 여기서는 양자화를 하지 않는다. 이 FP32 ONNX 가 _2 (ModelOpt 가 Q/DQ 삽입) 와 _4 (TensorRT 엔진) 의 입력이다.
- 검증: onnx checker + ONNX Runtime(CPU) 출력 == PyTorch(CPU) 출력 (test 이미지 2장, 클래스별 불일치 픽셀 수)
        + PyTorch deploy 모델의 test set 정확도 (학습 스크립트 summary 와 같은 값이 나와야 함: deploy 경로의 /255·정규화·임계값이 맞다는 뜻)

파이프라인:  _1_model2onnx.py  ->  _2_onnx_int8_ptq.py (ModelOpt INT8 PTQ)  ->  _3_int8_layer_check.py (점검)  ->  _4_onnx2trt.py

사용:
    python _1_model2onnx.py
"""
import cudnn_fix  # torch 보다 먼저: PATH 의 시스템 cuDNN 9.27 과 torch 번들 cuDNN 충돌 회피 (cudnn_fix.py 참고)
import os
import time
from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort
import torch

import eval_common as C
from eval_common import load_test_set, unpack_masks, metrics, print_accuracy_table, iou_summary, fmt_bytes
from model.unet_convnext import UNet_ConvNeXt


def check_training_config():
    """학습 스크립트의 상수와 eval_common 의 상수가 같은지 대조. 다르면 전처리/클래스가 어긋나므로 중단. 반환: 백본 이름"""
    try:
        import train_test_fp32_pcb as T
    except Exception as e:
        print(f"[warn] 학습 스크립트 import 실패, 상수 대조 생략: {type(e).__name__}: {e}")
        return "convnext_base"
    pairs = [("IMG_SIZE", tuple(T.IMG_SIZE), tuple(C.IMG_SIZE)), ("IN_CHANNELS", T.IN_CHANNELS, C.IN_CHANNELS),
             ("CLASSES", list(T.CLASSES), list(C.CLASSES)), ("THRESH", T.THRESH, C.THRESH)]
    bad = [f"{k}: 학습 {a} != eval_common {b}" for k, a, b in pairs if a != b]
    if bad:
        raise SystemExit("[config] eval_common.py 상수가 학습 스크립트와 다름 -> eval_common.py 를 맞출 것\n  " + "\n  ".join(bad))
    print(f"[config] 학습 스크립트와 일치: IMG_SIZE {C.IMG_SIZE}, IN_CHANNELS {C.IN_CHANNELS}, CLASSES {C.CLASSES}, "
          f"THRESH {C.THRESH}, encoder {T.ENCODER_NAME}")
    return T.ENCODER_NAME


def main():
    encoder_name = check_training_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = False          # GPU 기준값을 CPU/ORT 와 비교하므로 TF32 끔
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"[export] {C.CKPT_PATH} -> {C.ONNX_FP32}  (device {device})")

    # ---------- 1. 모델 (deploy 그래프) + 학습 가중치 ----------
    model = UNet_ConvNeXt(n_channels=C.IN_CHANNELS, n_classes=C.N_CLASSES, multilabel=True,
                          deploy=True, deploy_mask=True, deploy_thresh=C.THRESH,
                          encoder_name=encoder_name, drop_path_rate=0.0)
    state = torch.load(C.CKPT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)        # 정규화/비트 버퍼는 non-persistent 라 state_dict 에 없음 (모델이 만듦)
    model.eval().to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[model] UNet_ConvNeXt({encoder_name}) params {n_param:,} | state_dict tensors {len(state)} | "
          f"logit threshold {model.logit_thresh:.4f} (= logit(THRESH))")

    # ---------- 2. test set 으로 PyTorch deploy 출력 (정확도 확인 + ORT 비교 기준) ----------
    images, gts, paths = load_test_set()             # uint8 (N,3,H,W), bool (N,K,H,W)
    print(f"[data] test {len(images)} 장 {images.shape[1:]} uint8 BGR, GT {gts.shape[1:]} bool")
    t0 = time.perf_counter()
    outs = []
    with torch.no_grad():
        for i in range(len(images)):
            outs.append(model(torch.from_numpy(images[i:i + 1]).to(device)).cpu().numpy())
    ref = np.concatenate(outs)                       # uint8 (N,1,H,W)
    print(f"[torch] deploy output {ref.dtype} {ref.shape}, 값 {sorted(np.unique(ref).tolist())} (비트 패킹), "
          f"{(time.perf_counter() - t0) / len(images) * 1e3:.0f} ms/img ({device}, TF32 off)")
    m = metrics(unpack_masks(ref), gts)
    print_accuracy_table([("torch deploy", str(device), m)], len(images))
    print(f"[torch] {iou_summary(m)}  <- 학습 스크립트 summary 의 pixel-level IoU 와 같아야 함")

    # ---------- 3. ONNX export (dynamic batch, H/W 고정). torch>=2.9 는 dynamo exporter 가 기본이라 TorchScript exporter 명시 ----------
    dummy = torch.from_numpy(images[:1]).to(device)
    t0 = time.perf_counter()
    torch.onnx.export(
        model, dummy, C.ONNX_FP32,
        input_names=[C.INPUT_NAME], output_names=[C.OUTPUT_NAME],
        export_params=True, opset_version=C.OPSET, do_constant_folding=True,
        dynamic_axes={C.INPUT_NAME: {0: "batch_size"}, C.OUTPUT_NAME: {0: "batch_size"}},
        dynamo=False,
    )
    print(f"[export] done {time.perf_counter() - t0:.0f}s, {fmt_bytes(C.ONNX_FP32)}")

    # ---------- 4. 검증: checker + 그래프 요약 + ONNX Runtime(CPU) vs PyTorch ----------
    g = onnx.load(C.ONNX_FP32)
    onnx.checker.check_model(g)
    ops = Counter(n.op_type for n in g.graph.node)
    io = {v.name: (onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type),
                   [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]) for v in list(g.graph.input) + list(g.graph.output)}
    print(f"[onnx] nodes {len(g.graph.node)}, opset {[o.version for o in g.opset_import if o.domain in ('', 'ai.onnx')]}, I/O {io}")
    print(f"[onnx] ops: {dict(sorted(ops.items(), key=lambda kv: -kv[1]))}")
    print(f"[onnx] Q/DQ 노드: {ops.get('QuantizeLinear', 0)} / {ops.get('DequantizeLinear', 0)}  (FP32 그래프이므로 0 이 정상. INT8 은 _2 가 넣음)")

    n_cmp = min(2, len(images))
    sess = ort.InferenceSession(C.ONNX_FP32, providers=["CPUExecutionProvider"])
    t0 = time.perf_counter()
    out = sess.run(None, {C.INPUT_NAME: images[:n_cmp]})[0]       # batch 2 로 dynamic batch 확인
    print(f"[ort cpu] batch {n_cmp}: output {out.dtype} {out.shape}, {(time.perf_counter() - t0) / n_cmp:.1f} s/img")
    a, b = unpack_masks(out), unpack_masks(ref[:n_cmp])
    diff = {c: int((a[:, i] != b[:, i]).sum()) for i, c in enumerate(C.CLASSES)}
    total = int(np.prod(a.shape[2:])) * n_cmp
    print(f"[check] ORT(CPU) vs torch({device}) 마스크 불일치 픽셀 (test {n_cmp}장, 클래스별 {total:,} px 중): {diff}  "
          f"(경계의 소수 픽셀만 달라야 정상. fp32 연산 순서 차이)")
    if max(diff.values()) > total * 1e-3:
        raise SystemExit("[FAIL] ORT 와 torch 출력 차이가 너무 큼 (0.1% 초과). export 를 확인할 것")
    print(f"[OK] FP32 deploy ONNX: {C.ONNX_FP32}  -> 다음: python _2_onnx_int8_ptq.py")


if __name__ == "__main__":
    main()
