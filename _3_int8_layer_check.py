r"""
_3_int8_layer_check.py : INT8 변환 후 "어디가 INT8 로 바뀌었는지" 확인   (R6 UNet_ConvNeXt)

(1) ONNX 레벨 (기본): _2_onnx_int8_ptq.py 가 만든 Q/DQ ONNX 를 분석  -> ModelOpt 가 어디에 Q/DQ 를 넣었는가
    - Conv / ConvTranspose / MatMul / Gemm (가중치 있는 연산) 마다
        activation 입력이 DequantizeLinear 에서 오는지  -> INT8 activation
        weight 가 INT8 initializer + DequantizeLinear 인지 -> INT8 weight (per-channel / per-tensor, axis)
        아니면 입력 dtype (FP16 / FP32) 을 표시
    - 영역(ConvNeXt 백본 stem / stage 0~3, 경량 가지 inc/down1, 디코더 up1~up5/outc) × op 종류별 집계
    - 나머지 노드(LayerNormalization, Erf, Add, Resize ...)는 입력 dtype 별로 집계 (이들은 INT8 이 아니라 fp16 으로 실행됨)
(2) TensorRT 엔진 레벨 (옵션 --engine): 빌드된 엔진이 실제로 어떤 정밀도의 커널로 실행되는지 engine inspector 로 출력
    - 엔진을 profiling_verbosity=DETAILED 로 빌드해야 레이어 정보가 나옴 (_4 가 그렇게 빌드)
    - TensorRT 는 Q/DQ 를 Conv/MatMul 에 융합하고 LayerNorm/GELU 등은 fp16 커널로 두므로, ONNX 의 Q/DQ 개수와 엔진 레이어 수는 다르다

결과는 콘솔 + 리포트 json/html 의 "3. 모델 구조 중 INT8 / FP16 / FP32 위치" (eval_common.save_report -> report_html.py; --no-report 로 끔)

사용:
    python _3_int8_layer_check.py                          # 기본 경로의 INT8 ONNX 분석
    python _3_int8_layer_check.py --onnx path.onnx         # 다른 ONNX
    python _3_int8_layer_check.py --engine path.engine     # TensorRT 엔진 레이어 정밀도 (기본: _4 의 INT8 엔진)
    python _3_int8_layer_check.py --engine path.engine --summary   # 엔진은 집계만 (레이어 전체 목록 생략)
"""
import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict

import onnx
from onnx import TensorProto

from eval_common import ONNX_INT8, ENGINE_INT8, save_report

WEIGHTED_OPS = ("Conv", "ConvTranspose", "MatMul", "Gemm")
SKIP_OPS = ("QuantizeLinear", "DequantizeLinear", "Constant", "Cast", "Identity")

DTYPE_NAME = {
    TensorProto.FLOAT: "FP32", TensorProto.FLOAT16: "FP16", TensorProto.BFLOAT16: "BF16",
    TensorProto.INT8: "INT8", TensorProto.UINT8: "UINT8", TensorProto.INT32: "INT32", TensorProto.INT64: "INT64",
    TensorProto.BOOL: "BOOL", TensorProto.DOUBLE: "FP64",
}

# 영역 판정 순서 (노드 이름 접두어). torch export 이름: /encoder/stem_0/Conv, /encoder/stages_2/blocks/blocks.5/mlp/fc1/MatMul, /up3/reduce/Conv ...
REGIONS = [
    (r"^/encoder/stem", "백본 stem (1/4)"),
    (r"^/encoder/stages_0", "백본 stage 0 (1/4, 128ch)"),
    (r"^/encoder/stages_1", "백본 stage 1 (1/8, 256ch)"),
    (r"^/encoder/stages_2", "백본 stage 2 (1/16, 512ch)"),
    (r"^/encoder/stages_3", "백본 stage 3 (1/32, 1024ch)"),
    (r"^/encoder", "백본 기타"),
    (r"^/(inc|down1)/", "경량 가지 inc/down1 (1/1, 1/2)"),
    (r"^/up[1-5]/", "디코더 up1~up5"),
    (r"^/outc", "디코더 outc"),
]
REGION_ORDER = [r for _, r in REGIONS] + ["기타"]           # 기타 = 입력 정규화 (/255, mean/std), 출력 임계값·비트 패킹
OP_LABEL = {"Erf": "Erf (GELU)", "LayerNormalization": "LayerNorm", "Resize": "Resize (upsample)"}


def _prec_of(d):
    """dtype 이름 -> 'FP16' / 'FP32' / None ('INT8 (Q/DQ)', INT64, BOOL, '?' 등은 None)"""
    d = str(d)
    return "FP16" if d.startswith("FP16") else "FP32" if d.startswith("FP32") else None


def region_of(name):
    for pat, reg in REGIONS:
        if re.search(pat, name):
            return reg
    return "기타"


def region_of_any(name):
    """TensorRT 융합 레이어 이름 ('PWN(/encoder/stages_0/.../Div, ...)', 'A + B' 등 노드 이름 여러 개 포함) 에서 처음 매칭되는 영역"""
    for pat, reg in REGIONS:
        if re.search(pat.lstrip("^"), name):
            return reg
    return "기타"


def role_of(name, op):
    """노드 이름으로 역할 라벨 (ConvNeXt 블록 구성요소 등)"""
    if "conv_dw" in name:
        return "dwconv 7x7"
    if "/mlp/fc1" in name:
        return "Linear C->4C"
    if "/mlp/fc2" in name:
        return "Linear 4C->C"
    if "downsample" in name:
        return "downsample 2x2 s2"
    if "stem" in name:
        return "stem 4x4 s4"
    if "/reduce/" in name:
        return "1x1 reduce"
    if "/lateral/" in name:
        return "1x1 lateral(skip)"
    if "/conv/conv/" in name or re.search(r"/(inc|down1)/conv", name):
        return "3x3 conv"
    if "outc" in name:
        return "1x1 outc"
    return op


# ======================================================
# (1) ONNX 분석
# ======================================================
def analyze_onnx(path, write_report=True):
    model = onnx.load(path)
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:  # shape inference 가 실패해도 dtype 은 initializer / graph io 로 최대한 복원
        print(f"[warn] shape inference failed: {e}")
    g = model.graph

    inits = {t.name: t for t in g.initializer}
    dtype = {t.name: t.data_type for t in g.initializer}
    for v in list(g.input) + list(g.output) + list(g.value_info):
        if v.type.HasField("tensor_type"):
            dtype[v.name] = v.type.tensor_type.elem_type
    producer = {}
    for n in g.node:
        for o in n.output:
            producer[o] = n

    def dname(t):
        return DTYPE_NAME.get(dtype.get(t), "?")

    def dq_info(dq_node):
        """DequantizeLinear 노드의 (양자화 dtype, per-channel 여부, axis, 채널 수)"""
        x, scale = dq_node.input[0], dq_node.input[1]
        qdtype = dname(x)
        n_scale = 1
        if scale in inits:
            dims = list(inits[scale].dims)
            n_scale = int(math.prod(dims)) if dims else 1
        axis = next((a.i for a in dq_node.attribute if a.name == "axis"), None)
        return qdtype, n_scale > 1, axis, n_scale

    rows = []
    other = defaultdict(Counter)
    reg_prec = defaultdict(lambda: {"INT8": 0, "FP16": 0, "FP32": 0})   # 영역 -> 정밀도별 연산 개수 (리포트 차트)
    reg_ops = defaultdict(lambda: defaultdict(Counter))                   # 영역 -> 정밀도 -> 연산 라벨별 개수
    n_skipped = 0                                                         # dtype 이 FP16/FP32 가 아닌 노드 (shape/제어 연산)
    n_q = sum(n.op_type == "QuantizeLinear" for n in g.node)
    n_dq = sum(n.op_type == "DequantizeLinear" for n in g.node)

    for n in g.node:
        if n.op_type in SKIP_OPS:
            continue
        if n.op_type in WEIGHTED_OPS:
            act_in = n.input[0]
            w_in = n.input[1] if len(n.input) > 1 else None

            src = producer.get(act_in)
            if src is not None and src.op_type == "DequantizeLinear":
                qd, _, _, _ = dq_info(src)
                act, act_q = f"{qd} (Q/DQ)", qd == "INT8"
            else:
                act, act_q = dname(act_in), False

            w_q = False
            if w_in is None:
                weight = "-"
            else:
                wsrc = producer.get(w_in)
                if wsrc is not None and wsrc.op_type == "DequantizeLinear":
                    qd, per_ch, axis, n_scale = dq_info(wsrc)
                    weight = f"{qd} {'per-channel' if per_ch else 'per-tensor'}" + (f"(axis={axis}, {n_scale}ch)" if per_ch else "")
                    w_q = qd == "INT8"
                elif w_in in inits:
                    weight = dname(w_in)
                else:
                    weight = dname(w_in) + " (activation)"      # MatMul 의 두 입력이 모두 activation 인 경우
            out = dname(n.output[0])
            if act_q and w_q:
                verdict = "INT8"
            elif w_q or act_q:
                verdict = "PARTIAL"
            else:
                verdict = act if act != "?" else "UNKNOWN"
            name = n.name or n.output[0]
            reg, role = region_of(name), role_of(name, n.op_type)
            rows.append((name, n.op_type, reg, role, act, weight, out, verdict))
            prec = "INT8" if verdict == "INT8" else (_prec_of(out) or _prec_of(act) or "FP16")
            reg_prec[reg][prec] += 1
            reg_ops[reg][prec][f"{n.op_type} {role}" if role != n.op_type else n.op_type] += 1
        else:
            d = dname(n.input[0]) if n.input else "-"
            other[n.op_type][d] += 1
            prec = _prec_of(d)
            if prec is None:                                  # INT64/BOOL 입력 (shape, 제어) 또는 dtype 미상 -> 정밀도 집계 제외
                n_skipped += 1
                continue
            reg = region_of(n.name or (n.output[0] if n.output else ""))
            reg_prec[reg][prec] += 1
            reg_ops[reg][prec][OP_LABEL.get(n.op_type, n.op_type)] += 1

    print("=" * 150)
    print(f"[ONNX] {path}")
    print(f"       nodes: {len(g.node)} | QuantizeLinear: {n_q} | DequantizeLinear: {n_dq} | opset: {[o.version for o in model.opset_import if o.domain in ('', 'ai.onnx')]}")
    print("-" * 150)
    print(f"{'layer':58s}{'op':8s}{'role':18s}{'activation':14s}{'weight':30s}{'out':6s}{'=> runs as':10s}")
    print("-" * 150)
    for name, op, reg, role, act, weight, out, verdict in rows:
        print(f"{name[:57]:58s}{op:8s}{role[:17]:18s}{act:14s}{weight[:29]:30s}{out:6s}{verdict:10s}")
    print("-" * 150)

    by_verdict = Counter(v for *_, v in rows)
    print(f"weighted layers: {len(rows)} -> " + ", ".join(f"{k}: {v}" for k, v in sorted(by_verdict.items())))
    not_int8 = [(name, op, role, v) for name, op, reg, role, _, _, _, v in rows if v != "INT8"]
    if not_int8:
        print("not INT8:")
        for name, op, role, v in not_int8:
            print(f"  - {name} ({op}, {role}): {v}")

    # 영역 × op 집계
    agg = defaultdict(lambda: Counter())
    for name, op, reg, role, _, _, _, v in rows:
        agg[(reg, op)][v] += 1
    sum_rows = []
    for reg in REGION_ORDER:
        for op in WEIGHTED_OPS:
            c = agg.get((reg, op))
            if c:
                tot = sum(c.values())
                sum_rows.append([reg, op if op != "MatMul" else "MatMul (Linear)", tot, c.get("INT8", 0), tot - c.get("INT8", 0),
                                 ", ".join(f"{k} {v}" for k, v in sorted(c.items()) if k != "INT8") or "-"])
    print("by region x op (weighted layers):")
    print(f"  {'region':34s}{'op':17s}{'total':>6s}{'INT8':>6s}{'other':>6s}  other detail")
    for reg, op, tot, i8, oth, detail in sum_rows:
        print(f"  {reg:34s}{op:17s}{tot:6d}{i8:6d}{oth:6d}  {detail}")
    print("other nodes by input dtype (Q/DQ 가 안 붙는 연산. fp16 이면 TensorRT 에서도 fp16 커널):")
    for op, cnt in sorted(other.items()):
        print(f"  - {op:20s} " + ", ".join(f"{d}: {c}" for d, c in sorted(cnt.items())))
    print(f"by region x precision (연산 노드 전체. Q/DQ·Cast·Constant 와 dtype 이 FP16/FP32 가 아닌 노드 {n_skipped} 개 제외):")
    for reg in REGION_ORDER:
        if reg in reg_prec:
            c = reg_prec[reg]
            print(f"  - {reg:34s} INT8 {c['INT8']:4d}   FP16 {c['FP16']:4d}   FP32 {c['FP32']:4d}")
    print("=" * 150)

    if write_report:
        regions = [{"name": reg, **reg_prec[reg], "ops": {k: dict(v) for k, v in reg_ops[reg].items()}}
                   for reg in REGION_ORDER if reg in reg_prec]
        save_report("precision_onnx", {
            "file": path, "mb": os.path.getsize(path) / 2**20, "n_nodes": len(g.node), "n_q": n_q, "n_dq": n_dq,
            "weighted_total": len(rows), "weighted_int8": by_verdict.get("INT8", 0),
            "conv_int8": sum(1 for r in rows if r[1] in ("Conv", "ConvTranspose") and r[7] == "INT8"),
            "matmul_int8": sum(1 for r in rows if r[1] in ("MatMul", "Gemm") and r[7] == "INT8"),
            "n_skipped": n_skipped, "regions": regions,
        })
    return rows


# ======================================================
# (2) TensorRT 엔진 분석
# ======================================================
def analyze_engine(path, summary_only=False, write_report=True):
    """
    TensorRT 11 은 그래프 전체를 Myelin 으로 컴파일하므로 레이어 이름이 '__myl_...' (융합된 op 이름 나열) 이고, 원본 ONNX 노드는
    Metadata 의 '[ONNX Layer: ...]' 목록에 있다. 레이어 종류:
      correlation = Conv 커널, gemm = MatMul 커널 (입력 dtype 이 Int8 이면 INT8 연산),
      kgen = 생성된 융합 커널 (LayerNorm / GELU / 잔차 덧셈 / Transpose / Q·DQ 등). 입력이 Int8 이어도 내부 계산은 fp16/fp32 이고
             다시 Int8 로 양자화해 내보내는 경우가 많으므로 "입력→출력 dtype" 으로만 표시한다.
    """
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"failed to deserialize {path}")
    inspector = engine.create_engine_inspector()
    info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    layers = info.get("Layers", [])

    def tdt(t):
        """텐서 항목 -> dtype 이름 ('Int8' / 'Half' / 'Float' ...). TRT 11 은 'Datatype' 키, 예전 버전은 'Format/Datatype' 문자열"""
        if not isinstance(t, dict):
            return str(t)
        if t.get("Datatype"):
            return str(t["Datatype"])
        s = str(t.get("Format/Datatype", "?"))
        for k in ("Int8", "Half", "Float", "BFloat16", "FP8", "Int32", "UInt8", "Bool", "Int64"):
            if k in s:
                return k
        return s.split(" ")[0] or "?"

    def onnx_layers(L):
        return re.findall(r"\[ONNX Layer: ([^\]]+)\]", str(L.get("Metadata", "")))

    print("=" * 150)
    print(f"[TensorRT engine] {path}")
    print(f"       layers: {len(layers)} | TensorRT {trt.__version__}")
    if layers and isinstance(layers[0], str):
        print("       (엔진이 profiling_verbosity=DETAILED 로 빌드되지 않아 레이어 이름만 있음)")
        for name in layers:
            print(f"  - {name}")
        print("=" * 150)
        return None

    rows = []
    type_cnt, type_sig = Counter(), defaultdict(Counter)
    region_type = defaultdict(Counter)
    reg_prec = defaultdict(lambda: {"INT8": 0, "FP16": 0, "FP32": 0})   # 영역 -> 정밀도별 커널 개수 (리포트 차트)
    reg_ops = defaultdict(lambda: defaultdict(Counter))
    for L in layers:
        name = str(L.get("Name", "?"))
        ltype = str(L.get("LayerType", "?"))
        ins = [tdt(t) for t in L.get("Inputs") or []]
        outs = [tdt(t) for t in L.get("Outputs") or []]
        sig = f"{'/'.join(dict.fromkeys(ins)) or '-'} -> {'/'.join(dict.fromkeys(outs)) or '-'}"
        meta = onnx_layers(L)
        # 영역: Metadata 의 ONNX 노드 이름 중 Q/DQ 가 아닌 첫 노드 기준 (없으면 레이어 이름)
        core = [m for m in meta if "QuantizeLinear" not in m and "DequantizeLinear" not in m and "_cast_to_" not in m]
        reg = region_of_any(" ".join(core or meta) or name)
        type_cnt[ltype] += 1
        type_sig[ltype][sig] += 1
        region_type[reg][ltype] += 1
        rows.append((name, ltype, sig, core[:2], reg))
        if ltype != "shape_call":                         # shape 계산은 연산 커널이 아님
            if ltype in ("correlation", "gemm"):          # Conv / MatMul: 입력 dtype 이 곧 연산 정밀도
                prec = "INT8" if ins and ins[0] == "Int8" else ("FP32" if ins and ins[0] == "Float" else "FP16")
            else:                                         # kgen 등 융합 커널: Float 텐서가 있으면 FP32, 아니면 fp16 계산 (Int8 입출력이라도)
                prec = "FP32" if any(d == "Float" for d in ins + outs) else "FP16"
            reg_prec[reg][prec] += 1
            reg_ops[reg][prec][f"{ltype} {sig}"] += 1

    if not summary_only:
        print("-" * 150)
        print(f"{'layer':46s}{'type':13s}{'in -> out dtype':24s}{'region':30s}ONNX nodes (Q/DQ 제외, 앞 2개)")
        print("-" * 150)
        for name, ltype, sig, core, reg in rows:
            print(f"{name[:45]:46s}{ltype[:12]:13s}{sig[:23]:24s}{reg[:29]:30s}{', '.join(core)[:60]}")
        print("-" * 150)
    n_conv = type_cnt.get("correlation", 0); n_gemm = type_cnt.get("gemm", 0)
    i8_conv = sum(v for s, v in type_sig["correlation"].items() if s.startswith("Int8"))
    i8_gemm = sum(v for s, v in type_sig["gemm"].items() if s.startswith("Int8"))
    print(f"INT8 compute kernels: Conv(correlation) {i8_conv}/{n_conv}, MatMul(gemm) {i8_gemm}/{n_gemm}")
    print("layers by type (in -> out dtype):")
    for t, c in type_cnt.most_common():
        print(f"  - {t:14s} {c:4d}   " + ", ".join(f"{s} x{v}" for s, v in type_sig[t].most_common()))
    print("layers by region (Metadata 의 ONNX 노드 기준):")
    for reg in REGION_ORDER:
        if reg in region_type:
            print(f"  - {reg:34s} " + ", ".join(f"{t} {v}" for t, v in region_type[reg].most_common()))
    print("by region x precision (INT8 = Int8 입력 Conv/MatMul 커널, FP16 = 그 외 융합 커널, FP32 = Float 텐서를 다루는 커널):")
    for reg in REGION_ORDER:
        if reg in reg_prec:
            c = reg_prec[reg]
            print(f"  - {reg:34s} INT8 {c['INT8']:4d}   FP16 {c['FP16']:4d}   FP32 {c['FP32']:4d}")
    print("=" * 150)

    if write_report:
        regions = [{"name": reg, **reg_prec[reg], "ops": {k: dict(v) for k, v in reg_ops[reg].items()}}
                   for reg in REGION_ORDER if reg in reg_prec]
        save_report("precision_engine", {
            "file": path, "mb": os.path.getsize(path) / 2**20, "trt": trt.__version__, "n_layers": len(layers),
            "conv_int8": i8_conv, "conv_total": n_conv, "gemm_int8": i8_gemm, "gemm_total": n_gemm,
            "types": {t: dict(type_sig[t]) for t in type_cnt}, "regions": regions,
        })
    return type_sig


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="INT8 변환 결과에서 레이어별 정밀도 확인 (ONNX Q/DQ, TensorRT 엔진)")
    ap.add_argument("--onnx", nargs="?", const=ONNX_INT8, default=ONNX_INT8, help=f"분석할 Q/DQ ONNX (값 없이 주거나 생략하면 {ONNX_INT8})")
    ap.add_argument("--engine", nargs="?", const=ENGINE_INT8, default=None, help=f"TensorRT 엔진 파일 (값 없이 주면 {ENGINE_INT8})")
    ap.add_argument("--summary", action="store_true", help="엔진 분석 시 레이어 전체 목록 생략")
    ap.add_argument("--no-report", action="store_true", help="리포트 json/html 에 쓰지 않음")
    args = ap.parse_args()

    if args.engine:
        analyze_engine(args.engine, summary_only=args.summary, write_report=not args.no_report)
    else:
        analyze_onnx(args.onnx, write_report=not args.no_report)
