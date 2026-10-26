# PCB 패턴 검사 분할 모델: FP32 학습 → ONNX → ModelOpt INT8 → TensorRT 11 엔진 → 웹 서비스

ConvNeXt-Base 백본 UNet (`model/unet_convnext.py`) 으로 dataset_pcb (pcb / terminal / hole, multi-label) 를 학습하고,
TensorRT 11.4 INT8 엔진으로 배포한다. 입력은 uint8 BGR (B,3,1088,1344), 출력은 클래스별 마스크를 비트로 패킹한 uint8 (B,1,1088,1344)
(bit0 pcb, bit1 terminal, bit2 hole). /255, ImageNet 정규화, 임계값(sigmoid > 0.5) 은 모두 모델/엔진 안에 있다.

## 파일

| 파일 | 역할 |
|---|---|
| `model/unet_convnext.py` | R6 `UNet_ConvNeXt` (timm convnext_base 백본 + 경량 디코더, deploy 모드 포함) |
| `model/bitmask.py` | 비트 패킹 마스크 포맷 `unpack_bits` / `pack_bits` |
| `model/trt_runner.py` | TensorRT 엔진 실행기 `TrtRunner` (cupy 버퍼, execute_async_v3) |
| `model/docs/` | 모델 구조 설명 html |
| `images_dataset.py` | labelme json → 클래스별 마스크 Dataset |
| `train_test_fp32_pcb.py` | FP32 학습 / 평가 (`train`, `test`) |
| `eval_common.py` | 변환·배포 공통 설정(체크포인트·ONNX·엔진 경로, 크기, 클래스) 과 지표/측정/시각화 헬퍼 |
| `_1_model2onnx.py` | 체크포인트 → deploy FP32 ONNX |
| `_2_onnx_int8_ptq.py` | ModelOpt INT8 PTQ (Q/DQ 삽입 + calibration) |
| `_3_int8_layer_check.py` | 어디가 INT8 / FP16 / FP32 인지 (ONNX, 엔진) |
| `_4_onnx2trt.py` | TensorRT 엔진 빌드 (FP32 비교용 + INT8), 검증, 속도/GPU 메모리 |
| `_5_predict.py` | dense 체크포인트 / 엔진으로 폴더 추론: 모델별 오버레이 + 이미지별 지표 (라벨 있을 때), dense vs 엔진 비교 |
| `report_html.py` | `_2`/`_3`/`_4` 결과 json → 4항목 비교 리포트 html |
| `server_inference.py` | INT8 엔진 웹 서비스 (Flask + 공유메모리, batch 추론) |
| `client.py` | 서버 클라이언트 (폴더 이미지를 4장씩 전송, 오버레이/JSON 저장) |
| `cudnn_fix.py` | PATH 의 시스템 cuDNN 과 torch 번들 cuDNN 충돌 회피 (torch 보다 먼저 import) |

`dataset_pcb/` (train 22 장 / test 8 장, labelme json 라벨) 는 저장소에 포함되고, `output/` 등 학습·변환 산출물은 `.gitignore` 로 제외된다.

## 실행 순서

```powershell
python train_test_fp32_pcb.py train --num_epochs 500 --batch_size 2   # -> output/fp32_model_<날짜>/origin_model_best.pt
# eval_common.py 의 MODEL_DIR 를 그 폴더로 맞춘 뒤
python _1_model2onnx.py            # r6_fp32.onnx
python _2_onnx_int8_ptq.py         # r6_int8_qdq.onnx (ModelOpt, train 이미지로 calibration)
python _4_onnx2trt.py              # r6_fp32.engine, r6_int8.engine + 속도/정확도 비교
python _3_int8_layer_check.py --onnx --engine --summary   # 정밀도 위치 (선택)
python _5_predict.py               # dense + 엔진으로 test 폴더 추론, 모델별 오버레이 + 이미지별 지표, 비교표
python server_inference.py         # 웹 서비스 (127.0.0.1:5004)
python client.py                   # 다른 터미널: test 이미지를 4장씩 전송
```

TensorRT 11 은 calibrator 기반 INT8 이 없으므로 `_2` (ModelOpt 가 Q/DQ 를 넣음) 가 필수이고, `_1` + `_4` 만으로는 FP32 엔진이 나온다.
비교 리포트는 `output/<모델>/compare_fp32_vs_int8.html` 에 생성된다.

## 환경

Windows 11, RTX 5080, Python 3.10, torch 2.12 (cu128), timm, albumentations, onnx 1.21, onnxruntime-gpu 1.22, nvidia-modelopt 0.47,
tensorrt 11.4, cupy 14, flask, requests, opencv.
