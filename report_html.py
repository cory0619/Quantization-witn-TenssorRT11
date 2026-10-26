r"""
report_html.py : compare_fp32_vs_int8.json -> compare_fp32_vs_int8.html   (R6 UNet_ConvNeXt FP32 vs INT8 리포트)

_2_onnx_int8_ptq.py / _3_int8_layer_check.py / _4_onnx2trt.py 가 eval_common.save_report() 로 json 에 결과를 쌓을 때마다
이 모듈이 html 을 통째로 다시 만든다 (어느 스크립트를 먼저 돌려도 같은 결과, 아직 없는 항목은 자리만 표시).

내용은 네 가지만:
  1. 이미지당 처리 속도 (TensorRT 엔진, batch 1/4/8)            <- _4
  2. GPU 메모리 (activation / 프로세스 전용 / 엔진 파일)           <- _4
  3. 모델 구조 중 어디가 INT8 / FP16 / FP32 인지 (ONNX, 엔진)      <- _3
  4. 정확도 차이 (클래스별 IoU, mIoU, 마스크 일치율)              <- _4 (엔진 결과가 없으면 _2 의 ONNX Runtime 결과)
차트는 inline SVG (외부 라이브러리 없음, 오프라인에서 열림, 다크 모드 지원). 색은 모든 차트에서 INT8 파랑 / FP16 주황 / FP32 회색.

사용:  python report_html.py        # json 으로 html 만 다시 생성
"""
import html
import json
import math
import os
import sys
import time

# ------------------------------------------------------------------ 공통
PREC = [("INT8", "var(--int8)", "#fff"), ("FP16", "var(--fp16)", "#fff"), ("FP32", "var(--fp32)", "var(--ink)")]
ENG = [("fp32", "FP32 엔진", "var(--fp32)"), ("int8", "INT8 엔진", "var(--int8)")]
REGION_LABEL = {"기타": "입력 정규화 / 출력 임계값"}


def esc(s):
    return html.escape(str(s), quote=True)


def fnum(v, nd=1):
    return "-" if v is None else f"{v:,.{nd}f}"


def ticks(vmax, n=4):
    """0..vmax 를 덮는 보기 좋은 눈금 (step 1/2/2.5/5 x 10^k). 반환: (눈금 리스트, 축 최대)"""
    vmax = max(float(vmax), 1e-9)
    raw = vmax / n
    exp = 10 ** math.floor(math.log10(raw))
    step = exp
    for m in (1, 2, 2.5, 5, 10):
        step = m * exp
        if vmax / step <= n + 0.5:
            break
    top = math.ceil(vmax / step - 1e-9) * step
    k = int(round(top / step))
    return [i * step for i in range(k + 1)], top


def tick_label(t):
    return f"{t:g}" if t < 1000 else f"{t / 1000:g}K"


def rr(x, y, w, h, r=4):
    """오른쪽 끝만 둥근 막대 (기준선 쪽은 각짐)"""
    r = max(0.0, min(r, w / 2, h / 2))
    return (f"M{x:.1f},{y:.1f} h{w - r:.1f} a{r:.1f},{r:.1f} 0 0 1 {r:.1f},{r:.1f} v{h - 2 * r:.1f} "
            f"a{r:.1f},{r:.1f} 0 0 1 -{r:.1f},{r:.1f} h-{w - r:.1f} z")


def rl(x, y, w, h, r=4):
    """왼쪽 끝만 둥근 막대 (x = 왼쪽 끝)"""
    r = max(0.0, min(r, w / 2, h / 2))
    return (f"M{x + w:.1f},{y:.1f} h-{w - r:.1f} a{r:.1f},{r:.1f} 0 0 0 -{r:.1f},{r:.1f} v{h - 2 * r:.1f} "
            f"a{r:.1f},{r:.1f} 0 0 0 {r:.1f},{r:.1f} h{w - r:.1f} z")


# ------------------------------------------------------------------ SVG 차트
def svg_grouped_hbars(rows, series, unit, value_fmt=lambda v: f"{v:.1f}", label_w=96, plot_w=480,
                      bar_h=14, gap=2, group_gap=16):
    """
    가로 그룹 막대. rows: [(행 라벨, {series_key: 값})], series: [(key, 이름, 색)]. 값 라벨은 막대 끝.
    """
    n = len(series)
    group_h = n * bar_h + (n - 1) * gap
    top, bottom = 6, 30
    H = top + len(rows) * (group_h + group_gap) - group_gap + bottom
    W = label_w + plot_w + 96
    vals = [v for _, d in rows for v in d.values() if v is not None]
    tks, vtop = ticks(max(vals) if vals else 1)
    sx = lambda v: label_w + v / vtop * plot_w
    y0, y1 = top, H - bottom
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" role="img" aria-label="{esc(unit)}">']
    for t in tks:
        x = sx(t)
        out.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" class="grid"/>'
                   f'<text x="{x:.1f}" y="{y1 + 16}" class="tick" text-anchor="middle">{tick_label(t)}</text>')
    out.append(f'<line x1="{label_w}" y1="{y0}" x2="{label_w}" y2="{y1}" class="axis"/>')
    out.append(f'<text x="{W - 4}" y="{H - 4}" class="tick" text-anchor="end">{esc(unit)}</text>')
    y = top
    for label, d in rows:
        if label:
            out.append(f'<text x="{label_w - 8}" y="{y + group_h / 2:.1f}" class="lbl" text-anchor="end" dominant-baseline="middle">{esc(label)}</text>')
        for i, (key, sname, color) in enumerate(series):
            v = d.get(key)
            if v is None:
                continue
            by = y + i * (bar_h + gap)
            w = sx(v) - label_w
            out.append(f'<g><title>{esc(sname)}{(" · " + esc(label)) if label else ""}: {value_fmt(v)} {esc(unit)}</title>'
                       f'<rect x="{label_w}" y="{by - 2}" width="{max(w, 1) + 60:.1f}" height="{bar_h + 4}" fill="transparent"/>'
                       f'<path d="{rr(label_w, by, w, bar_h)}" fill="{color}"/>'
                       f'<text x="{sx(v) + 6:.1f}" y="{by + bar_h / 2:.1f}" class="val" dominant-baseline="middle">{value_fmt(v)}</text></g>')
        y += group_h + group_gap
    out.append("</svg>")
    return "\n".join(out)


def svg_stacked(rows, label_w=186, plot_w=380, bar_h=16, row_gap=10):
    """
    영역별 100% 누적 막대 (정밀도 구성 비율). rows: [(라벨, {"INT8": n, "FP16": n, "FP32": n}, 툴팁)].
    세그먼트 안 = 개수, 막대 끝 = 영역 전체 개수. 영역마다 연산 수가 4~409 로 크게 달라 절대 길이로 그리면 작은 영역이 안 보이므로 비율로 그린다.
    """
    totals = [sum(c.get(k, 0) for k, _, _ in PREC) for _, c, _ in rows]
    top, bottom = 6, 26
    H = top + len(rows) * (bar_h + row_gap) - row_gap + bottom
    W = label_w + plot_w + 56
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" role="img" aria-label="정밀도별 연산 비율">']
    y1 = H - bottom
    for pct in (0, 25, 50, 75, 100):
        x = label_w + plot_w * pct / 100
        out.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{y1}" class="{"axis" if pct == 0 else "grid"}"/>'
                   f'<text x="{x:.1f}" y="{y1 + 14}" class="tick" text-anchor="middle">{pct}%</text>')
    y = top
    for (label, c, tip), total in zip(rows, totals):
        out.append(f'<text x="{label_w - 8}" y="{y + bar_h / 2:.1f}" class="lbl" text-anchor="end" dominant-baseline="middle">{esc(label)}</text>')
        x = label_w
        scale = plot_w / max(total, 1)
        segs = [(k, color, tc, c.get(k, 0)) for k, color, tc in PREC if c.get(k, 0) > 0]
        out.append(f'<g><title>{esc(tip)}</title>')
        for j, (k, color, tc, cnt) in enumerate(segs):
            w = cnt * scale
            x0 = x + (2 if j > 0 else 0)
            w0 = w - (2 if j > 0 else 0)
            if w0 > 0.5:
                out.append(f'<rect x="{x0:.1f}" y="{y}" width="{w0:.1f}" height="{bar_h}" fill="{color}"/>')
                if w0 >= 26:
                    out.append(f'<text x="{x0 + w0 / 2:.1f}" y="{y + bar_h / 2:.1f}" class="inlbl" fill="{tc}" text-anchor="middle" dominant-baseline="middle">{cnt}</text>')
            x += w
        out.append(f'<text x="{label_w + plot_w + 8}" y="{y + bar_h / 2:.1f}" class="val" dominant-baseline="middle">{total}</text></g>')
        y += bar_h + row_gap
    out.append("</svg>")
    return "\n".join(out)


def svg_diverging(rows, unit="pp", label_w=90, plot_w=420, bar_h=14, row_gap=10):
    """0 을 가운데 둔 양방향 막대. rows: [(라벨, 값)]. 양수 파랑, 음수 빨강."""
    vmax = max([abs(v) for _, v in rows] + [1e-6])
    tks, vtop = ticks(vmax, 2)
    cx = label_w + plot_w / 2
    sx = lambda v: cx + v / vtop * (plot_w / 2)
    top, bottom = 6, 30
    H = top + len(rows) * (bar_h + row_gap) - row_gap + bottom
    W = label_w + plot_w + 70
    y0, y1 = top, H - bottom
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" role="img" aria-label="차이 ({esc(unit)})">']
    for t in tks:
        for v in ({0.0} if t == 0 else {t, -t}):
            x = sx(v)
            cls = "axis" if v == 0 else "grid"
            lab = "0" if v == 0 else f"{v:+g}"
            out.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" class="{cls}"/>'
                       f'<text x="{x:.1f}" y="{y1 + 16}" class="tick" text-anchor="middle">{lab}</text>')
    out.append(f'<text x="{W - 4}" y="{H - 4}" class="tick" text-anchor="end">{esc(unit)}</text>')
    y = top
    for label, v in rows:
        out.append(f'<text x="{label_w - 8}" y="{y + bar_h / 2:.1f}" class="lbl" text-anchor="end" dominant-baseline="middle">{esc(label)}</text>')
        w = abs(sx(v) - cx)
        if v >= 0:
            path, tx, anchor, color = rr(cx, y, w, bar_h), sx(v) + 6, "start", "var(--pos)"
        else:
            path, tx, anchor, color = rl(sx(v), y, w, bar_h), sx(v) - 6, "end", "var(--neg)"
        out.append(f'<g><title>{esc(label)}: {v:+.3f} {esc(unit)}</title>'
                   f'<rect x="{min(cx, sx(v)) - 4:.1f}" y="{y - 2}" width="{w + 8:.1f}" height="{bar_h + 4}" fill="transparent"/>'
                   f'<path d="{path}" fill="{color}"/>'
                   f'<text x="{tx:.1f}" y="{y + bar_h / 2:.1f}" class="val" text-anchor="{anchor}" dominant-baseline="middle">{v:+.2f}</text></g>')
        y += bar_h + row_gap
    out.append("</svg>")
    return "\n".join(out)


def legend(items):
    return '<div class="legend">' + "".join(f'<span><i style="background:{c}"></i>{esc(n)}</span>' for n, c in items) + "</div>"


def table(headers, rows, cls=""):
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tbl"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def tile(label, value, sub):
    return f'<div class="tile"><div class="l">{esc(label)}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'


def pending(text):
    return f'<p class="pending">{esc(text)}</p>'


# ------------------------------------------------------------------ 섹션
def sec_speed(trt):
    if not trt or not trt.get("speed"):
        return pending("_4_onnx2trt.py 실행 후 표시됩니다 (TensorRT 엔진 속도)."), None
    sp = trt["speed"]
    batches = sorted({r["batch"] for r in sp})
    get = lambda eng, b, k: next((r[k] for r in sp if r["engine"] == eng and r["batch"] == b), None)
    rows = [(f"batch {b}", {e: (get(e, b, "gpu_ms") / b if get(e, b, "gpu_ms") else None) for e, _, _ in ENG}) for b in batches]
    chart = svg_grouped_hbars(rows, ENG, "GPU 실행 ms / 장", plot_w=560)
    trows = []
    for b in batches:
        g32, g8, e32, e8 = get("fp32", b, "gpu_ms"), get("int8", b, "gpu_ms"), get("fp32", b, "e2e_ms"), get("int8", b, "e2e_ms")
        trows.append([b, fnum(g32 / b if g32 else None, 2), fnum(g8 / b if g8 else None, 2), f"×{g32 / g8:.2f}" if g32 and g8 else "-",
                      fnum(e32 / b if e32 else None, 2), fnum(e8 / b if e8 else None, 2), f"×{e32 / e8:.2f}" if e32 and e8 else "-"])
    n = trt.get("bench", {})
    note = (f"GPU 실행 = execute_async_v3 + 동기화 (입력이 이미 GPU 에 있음). 복사 포함 = pinned host→device 복사 + 실행 + device→host 복사. "
            f"warmup {n.get('warmup', '?')} / 측정 {n.get('iters', '?')} 회 평균, ms/장 = ms/batch ÷ batch.")
    body = (legend([(n_, c) for _, n_, c in ENG]) + chart
            + table(["batch", "FP32 GPU ms/장", "INT8 GPU ms/장", "속도비", "FP32 복사 포함", "INT8 복사 포함", "속도비"], trows, "num")
            + f'<p class="note">{esc(note)}</p>')
    # KPI 용
    bmax = max(batches)
    g32, g8 = get("fp32", bmax, "gpu_ms"), get("int8", bmax, "gpu_ms")
    kpi = None
    if g32 and g8:
        kpi = tile(f"GPU 실행 시간 / 장 (batch {bmax})", f"{g8 / bmax:.1f} ms",
                   f"FP32 {g32 / bmax:.1f} ms → <b>×{g32 / g8:.2f}</b> 빠름")
    return body, kpi


def sec_memory(trt):
    if not trt or not trt.get("engines"):
        return pending("_4_onnx2trt.py 실행 후 표시됩니다 (GPU 메모리)."), None
    e32, e8 = trt["engines"].get("fp32"), trt["engines"].get("int8")
    if not e8:
        return pending("INT8 엔진 결과가 없습니다."), None
    panels = []
    items = [("프로세스 GPU 메모리 (추론 중 최대)", "peak_mib", "MiB", 0, "작업관리자의 전용 GPU 메모리. CUDA context 포함"),
             ("엔진 activation 메모리 (max batch)", "activation_mib", "MiB", 0, "engine.device_memory_size_v2"),
             ("엔진 파일 크기", "size_mb", "MB", 1, "가중치 INT8 저장")]
    trows = []
    for title, key, unit, nd, hint in items:
        v32, v8 = (e32 or {}).get(key), e8.get(key)
        chart = svg_grouped_hbars([("", {"fp32": v32, "int8": v8})], ENG, unit, value_fmt=lambda v, nd=nd: f"{v:,.{nd}f}",
                                  label_w=8, plot_w=200, bar_h=14, gap=2)
        delta = f"<b>−{(1 - v8 / v32) * 100:.0f}%</b> (FP32 대비)" if v32 and v8 else ""
        panels.append(f'<div class="panel"><div class="pt">{esc(title)}</div>{chart}<div class="pd">{delta}</div></div>')
        trows.append([title, fnum(v32, nd), fnum(v8, nd), f"{v8 / v32 * 100:.0f}%" if v32 and v8 else "-", hint])
    body = (legend([(n_, c) for _, n_, c in ENG]) + '<div class="panels">' + "".join(panels) + "</div>"
            + table(["항목", "FP32 엔진", "INT8 엔진", "INT8 / FP32", "설명"], trows, "num")
            + f'<p class="note">CUDA context 만 있을 때 프로세스 GPU 메모리 {fnum(trt.get("base_mib"), 0)} MiB. '
              f'profile batch {e8.get("profile", "?")} (min/opt/max), 측정 batch {", ".join(str(b) for b in trt.get("bench", {}).get("batches", []))}.</p>')
    kpi = None
    if e32 and e32.get("peak_mib") and e8.get("peak_mib"):
        kpi = tile("GPU 메모리 (추론 중 최대)", f"{e8['peak_mib']:,.0f} MiB",
                   f"FP32 {e32['peak_mib']:,.0f} MiB → <b>−{(1 - e8['peak_mib'] / e32['peak_mib']) * 100:.0f}%</b>")
    return body, kpi


def _prec_rows(section):
    rows, trows = [], []
    for r in section.get("regions", []):
        name = REGION_LABEL.get(r["name"], r["name"])
        ops = r.get("ops", {})
        parts = []
        for k, _, _ in PREC:
            if r.get(k):
                detail = ", ".join(f"{o} {c}" for o, c in sorted(ops.get(k, {}).items(), key=lambda kv: -kv[1]))
                parts.append(f"{k} {r[k]}: {detail}")
        rows.append((name, {k: r.get(k, 0) for k, _, _ in PREC}, f"{name}\n" + "\n".join(parts)))
        trows.append([esc(name)] + [str(r.get(k, 0)) for k, _, _ in PREC]
                     + ["<br>".join(f"<b>{k}</b> " + esc(", ".join(f"{o} {c}" for o, c in sorted(ops.get(k, {}).items(), key=lambda kv: -kv[1])))
                        for k, _, _ in PREC if r.get(k)) or "-"])
    return rows, trows


def sec_precision(po, pe):
    if not po and not pe:
        return pending("_3_int8_layer_check.py --onnx / --engine 실행 후 표시됩니다 (레이어별 정밀도)."), None
    cols, tables = [], []
    tot = lambda s, k: sum(r.get(k, 0) for r in s.get("regions", []))
    if po:
        rows, trows = _prec_rows(po)
        cols.append(f'<div><div class="ct">ONNX 노드 (ModelOpt Q/DQ 삽입 후) · INT8 {tot(po, "INT8")} / FP16 {tot(po, "FP16")} / FP32 {tot(po, "FP32")}</div>'
                    + svg_stacked(rows) + "</div>")
        tables.append(f'<details><summary>ONNX 노드 내역 (영역 × 정밀도 × 연산)</summary>'
                      + table(["영역", "INT8", "FP16", "FP32", "연산 (개수)"], trows, "num ops") + "</details>")
    else:
        cols.append('<div><div class="ct">ONNX 노드</div>' + pending("_3_int8_layer_check.py --onnx 실행 후 표시") + "</div>")
    if pe:
        rows, trows = _prec_rows(pe)
        cols.append(f'<div><div class="ct">TensorRT 엔진 커널 · INT8 {tot(pe, "INT8")} / FP16 {tot(pe, "FP16")} / FP32 {tot(pe, "FP32")}</div>'
                    + svg_stacked(rows) + "</div>")
        tables.append(f'<details><summary>엔진 커널 내역 (영역 × 정밀도 × 커널 종류, 입력→출력 dtype)</summary>'
                      + table(["영역", "INT8", "FP16", "FP32", "커널 (개수)"], trows, "num ops") + "</details>")
    else:
        cols.append('<div><div class="ct">TensorRT 엔진 커널</div>' + pending("_4 로 엔진을 만든 뒤 _3 --engine 실행 후 표시") + "</div>")
    note = ("INT8 = activation 과 weight 양쪽에 Q/DQ 가 붙은 Conv / MatMul (TensorRT 가 Int8 입력 Conv·MatMul 커널로 실행). "
            "FP16 = 그 외 연산 (LayerNorm, GELU, 잔차 덧셈, Transpose, Resize, 입력 정규화, 출력 임계값; 엔진에서는 Int8 을 받아 fp16 으로 계산하는 융합 커널). "
            "FP32 = float32 텐서를 다루는 연산. 막대 = 영역 안 연산 개수의 정밀도 비율, 오른쪽 숫자 = 영역 전체 개수. "
            "개수 기준이라 FP16 이 많아 보이지만 연산량(MAC)은 거의 전부 INT8 쪽 Conv/MatMul 에 있다.")
    body = (legend([(k, c) for k, c, _ in PREC]) + '<div class="cols">' + "".join(cols) + "</div>"
            + "".join(tables) + f'<p class="note">{esc(note)}</p>')
    kpi = None
    if po and po.get("weighted_total"):
        extra = ""
        if pe:
            extra = f", 엔진 Int8 커널 Conv {pe.get('conv_int8', '?')}/{pe.get('conv_total', '?')} · MatMul {pe.get('gemm_int8', '?')}/{pe.get('gemm_total', '?')}"
        kpi = tile("INT8 로 바뀐 가중치 연산", f"{po['weighted_int8']} / {po['weighted_total']}",
                   f"Conv {po.get('conv_int8', '?')} + MatMul {po.get('matmul_int8', '?')}{esc(extra)}")
    return body, kpi


def sec_accuracy(trt, ort_, classes):
    src, acc, agree, label = None, None, None, ""
    if trt and trt.get("accuracy", {}).get("int8") and trt["accuracy"].get("fp32"):
        src, acc, agree, label = "trt", trt["accuracy"], trt.get("agreement_fp32_int8"), "TensorRT 엔진"
    elif ort_ and ort_.get("accuracy", {}).get("int8") and ort_["accuracy"].get("fp32"):
        src, acc, agree, label = "ort", ort_["accuracy"], ort_.get("agreement"), "ONNX Runtime (엔진 결과 없음)"
    if not src:
        return pending("_4_onnx2trt.py (또는 _2_onnx_int8_ptq.py) 실행 후 표시됩니다 (정확도)."), None
    a32, a8 = acc["fp32"], acc["int8"]
    classes = a32.get("classes", classes)
    rows = [(c, (a8["px"][c]["iou"] - a32["px"][c]["iou"]) * 100) for c in classes]
    rows.append(("mIoU", (a8["miou_px"] - a32["miou_px"]) * 100))
    chart = svg_diverging(rows, "pp (INT8 − FP32, IoU × 100)")
    trows = []
    for c in classes:
        p32, p8 = a32["px"][c], a8["px"][c]
        trows.append([c, f"{p32['iou']:.4f}", f"{p8['iou']:.4f}", f"{(p8['iou'] - p32['iou']) * 100:+.2f}",
                      f"{p32['f1']:.4f}", f"{p8['f1']:.4f}", f"{agree[c] * 100:.3f}%" if agree and c in agree else "-"])
    trows.append(["<b>mIoU</b>", f"<b>{a32['miou_px']:.4f}</b>", f"<b>{a8['miou_px']:.4f}</b>", f"<b>{(a8['miou_px'] - a32['miou_px']) * 100:+.2f}</b>",
                  "", "", f"{agree['all'] * 100:.3f}% (세 비트 모두)" if agree and "all" in agree else "-"])
    n_img = a32.get("n_images", "?")
    note = (f"{label}, test {n_img} 장, pixel-level (전체 픽셀 TP/FP/FN 합산). 임계값 sigmoid > 0.5 는 모델 안. "
            "마스크 일치율 = FP32 와 INT8 출력 마스크가 같은 픽셀 비율. 1 pp = IoU 0.01.")
    body = (chart + table(["클래스", "FP32 IoU", "INT8 IoU", "차이 (pp)", "FP32 F1", "INT8 F1", "마스크 일치율"], trows, "num")
            + f'<p class="note">{esc(note)}</p>')
    d = (a8["miou_px"] - a32["miou_px"]) * 100
    kpi = tile("mIoU 변화 (INT8 − FP32)", f"{d:+.2f} pp", f"FP32 {a32['miou_px']:.4f} → INT8 {a8['miou_px']:.4f}" + ("" if src == "trt" else " (ONNX Runtime)"))
    return body, kpi


# ------------------------------------------------------------------ 문서
CSS = """
:root { color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
        --border:rgba(11,11,11,.10); --int8:#2a78d6; --fp16:#eb6834; --fp32:#9e9c95; --pos:#2a78d6; --neg:#e34948; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7;
        --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10); --int8:#3987e5; --fp16:#d95926; --fp32:#767470; --pos:#3987e5; --neg:#e66767; } }
:root[data-theme="dark"] { color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
        --border:rgba(255,255,255,.10); --int8:#3987e5; --fp16:#d95926; --fp32:#767470; --pos:#3987e5; --neg:#e66767; }
* { box-sizing:border-box; }
body { margin:0; padding:24px 16px 48px; background:var(--page); color:var(--ink); font:14px/1.55 system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif; }
main { max-width:1120px; margin:0 auto; }
h1 { font-size:1.45em; margin:0 0 4px; }
h2 { font-size:1.1em; margin:0 0 10px; }
.meta { color:var(--ink2); font-size:13px; margin:0 0 18px; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px 20px; margin:14px 0; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; margin:0 0 4px; }
.tile { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:12px 16px; }
.tile .l { color:var(--ink2); font-size:13px; } .tile .v { font-size:28px; font-weight:600; line-height:1.25; margin:2px 0; } .tile .s { color:var(--ink2); font-size:13px; }
.legend { display:flex; flex-wrap:wrap; gap:14px; color:var(--ink2); font-size:13px; margin:0 0 6px; }
.legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:-1px; }
.cols { display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:18px; }
.ct { color:var(--ink2); font-size:13px; margin:0 0 4px; }
.panels { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:16px; }
.panel .pt { color:var(--ink2); font-size:13px; } .panel .pd { color:var(--ink2); font-size:13px; }
svg { display:block; overflow:visible; } svg text { font-family:inherit; font-size:12px; fill:var(--ink2); }
svg .lbl { fill:var(--ink); } svg .val { fill:var(--ink2); font-variant-numeric:tabular-nums; } svg .tick { fill:var(--muted); font-size:11px; }
svg .inlbl { font-size:11px; font-weight:600; } svg .grid { stroke:var(--grid); stroke-width:1; } svg .axis { stroke:var(--axis); stroke-width:1; }
.tbl { overflow-x:auto; margin:10px 0 4px; } table { border-collapse:collapse; font-size:13px; }
th, td { border-bottom:1px solid var(--grid); padding:4px 10px; text-align:left; vertical-align:top; } th { color:var(--ink2); font-weight:600; }
table.num td:not(:first-child) { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
table.ops td:last-child { text-align:left; white-space:normal; min-width:320px; }
details { margin:8px 0; } summary { cursor:pointer; color:var(--ink2); font-size:13px; }
.note { color:var(--muted); font-size:12px; margin:8px 0 0; } .pending { color:var(--muted); font-style:italic; }
b { font-weight:600; }
"""


def render(data):
    env, trt, po, pe, ort_ = data.get("env", {}), data.get("trt"), data.get("precision_onnx"), data.get("precision_engine"), data.get("ort")
    classes = env.get("classes") or ["pcb", "terminal", "hole"]
    s_speed, k_speed = sec_speed(trt)
    s_mem, k_mem = sec_memory(trt)
    s_prec, k_prec = sec_precision(po, pe)
    s_acc, k_acc = sec_accuracy(trt, ort_, classes)

    meta = []
    if env:
        meta.append(f"{env.get('model', 'R6 UNet_ConvNeXt')} · {env.get('dataset', '')} test {env.get('n_test', '?')}장 "
                    f"{'×'.join(str(v) for v in env.get('img_size', []))} · 클래스 {' / '.join(classes)}")
        meta.append(f"ModelOpt {env.get('modelopt', '?')} INT8 PTQ (calibration {env.get('calib_method', '?')}, {env.get('n_calib', '?')}장, 나머지 {env.get('high_precision', 'fp16')})")
    if trt:
        meta.append(f"TensorRT {trt.get('trt', '?')} · {trt.get('gpu', env.get('gpu', '?'))} · driver {trt.get('driver', env.get('driver', '?'))}")
    elif env.get("gpu"):
        meta.append(f"{env['gpu']} · driver {env.get('driver', '?')}")
    stamps = {k: v.get("updated") for k, v in data.items() if isinstance(v, dict) and v.get("updated")}
    meta.append("갱신: " + ", ".join(f"{k} {v}" for k, v in stamps.items()))

    kpis = "".join(k for k in (k_speed, k_mem, k_prec, k_acc) if k)
    doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FP32 vs INT8 비교</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>FP32 vs INT8 비교 (R6 UNet_ConvNeXt, dataset_pcb)</h1>
<p class="meta">{"<br>".join(esc(m) for m in meta)}</p>
<div class="kpis">{kpis}</div>
<section class="card"><h2>1. 이미지당 처리 속도 (TensorRT 엔진)</h2>{s_speed}</section>
<section class="card"><h2>2. GPU 메모리</h2>{s_mem}</section>
<section class="card"><h2>3. 모델 구조 중 INT8 / FP16 / FP32 위치</h2>{s_prec}</section>
<section class="card"><h2>4. 정확도 차이</h2>{s_acc}</section>
<p class="note">생성 {time.strftime("%Y-%m-%d %H:%M")} · 원본 데이터 compare_fp32_vs_int8.json (_2 / _3 / _4 가 기록)</p>
</main>
</body>
</html>
"""
    return doc


def write_report_html(json_path, html_path=None):
    html_path = html_path or os.path.splitext(json_path)[0] + ".html"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(render(data))
    print(f"[html] -> {html_path}")
    return html_path


if __name__ == "__main__":
    from eval_common import REPORT_JSON, REPORT_HTML
    path = sys.argv[1] if len(sys.argv) > 1 else REPORT_JSON
    if not os.path.exists(path):
        raise SystemExit(f"{path} 가 없음. _2 / _3 / _4 를 먼저 실행")
    write_report_html(path, REPORT_HTML if len(sys.argv) <= 1 else None)
