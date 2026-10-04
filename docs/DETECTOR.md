# ApolloNet-Pose: детектор для Rockchip NPU и полный пайплайн обучения

Что здесь есть: архитектура модели (`src/dataopen/detector/model.py`), потери и назначение целей, аугментации, цикл обучения,
метрики с проверкой целей, экспорт в ONNX, INT8-квантование с калибровочным набором, постобработка (Python и C), бинарная
структура результата, бенчмарк и скрипты для платы. Всё запускается командой `dataopen detector ...`.

## 0. Честный статус

| Что | Статус |
|---|---|
| Модель, потери, обучение, ONNX-экспорт, INT8 (ONNX Runtime), постобработка, C-реализация, структура, калибровочный набор, IoU-согласие | **Реализовано и проверено тестами** (`tests/test_detector_*.py`, CI-job `detector`). Крошечная модель за 160 шагов на CPU учится на игрушечных данных (map50 ≈ 0.73), экспорт совпадает с PyTorch до 1e-7, ONNX работает в замкнутом цикле как базовая модель. |
| Конвертация в `.rknn`, точность INT8 на NPU, **latency < 4 мс, > 240 FPS** | **Не проверено.** Здесь нет платы, нет rknn-toolkit2 (он x86 Linux, от Rockchip). Есть скрипты конверсии и бенчмарка, которые вы запускаете у себя, и арифметическая оценка. |
| **mAP@0.5 > 0.90, OKS@0.5 > 0.85** на вашем val | **Неизвестно.** Нужны реальные данные и обучение на GPU. Инструмент проверки цели есть (`detector eval --require`), гарантии качества нет. |
| QAT (обучение с квантованием) | Не реализовано; это запасной путь, если PTQ потеряет больше допустимого (раздел 6). |

## 1. Требования и решения

| Требование | Решение | Почему |
|---|---|---|
| Вход 640×640×3 uint8 | ONNX с входом uint8 NHWC (для ORT/цикла) или float 0..255 NCHW (для RKNN); 1/255 свёрнут в первый conv | NPU квантует вход сам; нет лишних операций |
| Backbone «в духе HGNetV2-B2 ×0.75 или лучше» | **RepHG**: агрегирующие блоки HGNetV2, внутри RepVGG-свёртки (3×3+1×1+identity при обучении, **один 3×3 на плате**), только плотные свёртки и ReLU | B2×0.75 по порядку величины на десятки GFLOPs (D-FINE-M около 57 GFLOPs по публикации, не проверялось здесь), в 4 мс на NPU не укладывается. Depthwise плохо загружает NPU, SiLU/GELU/SE и attention плохо квантуются |
| Neck multi-level | **RepBiFPN**: top-down + bottom-up по P3/P4/P5, обучаемые неотрицательные веса слияния (запекаются в константы) | двунаправленный поток с дешёвым слиянием; nearest-Resize, Add, Mul |
| Head decoupled, без NMS | cls / box / keypoints раздельно; **два головы при обучении** (one-to-many top-10 и one-to-one top-1), на плате только one-to-one | как YOLOv10: плотная суперобучаемость + выход без дублей. DETR/D-FINE-декодер (attention, LayerNorm, softmax) для INT8 NPU плох |
| INT8, per-channel веса, per-tensor активации | сеть спроектирована под это (раздел 6); `quantize` ставит именно такую схему | |
| Rockchip RKNN | в графе только Conv, ReLU, Add, Mul, Concat, Resize(nearest) (тест это проверяет); exp/sigmoid/top-k вынесены на хост | |
| 12 точек + 2 класса | схема `shooter12` (docs/SCHEMAS.md), сигмы/веса OKS из неё | |
| > 240 FPS и < 4 мс | два профиля: **n** (≈1.2 GMACs) под латентность, **s** (≈2.3 GMACs) под точность/пропускную способность | раздел 7 |

**Что сильнее базового «dual-level детектора»:** (1) one-to-one/one-to-many без NMS вместо постобработки с порогами; (2) отдельная
**голова уточнения прицельной точки** (heatmap stride 4 + субклеточные смещения): голова размером 3–6 px не берётся
регрессией со stride 8 с нужной точностью; (3) веса OKS и сигмы берутся из схемы (голова ×3); (4) выборка кадров по весу
замкнутого цикла (`weight` из вердикта, сложные кадры видны чаще); (5) выходы подобраны под INT8 (box в лог-масштабе,
смещения в единицах 4×stride, логиты без softmax); (6) исправлена структура результата (раздел 3).

## 2. Архитектура

```
uint8 640x640x3 ─► stem (RepConv s2, s2) ─► P2 (s4)
                       │
        stage s8  ─ RepConv s2 + RepHG ×1 ──► P3 ┐
        stage s16 ─ RepConv s2 + RepHG ×1-2 ─► P4 ├─► RepBiFPN (1×) ─► N3 N4 N5 ─► Head ×3 уровня ─► p3 p4 p5
        stage s32 ─ RepConv s2 + RepHG ×1 ──► P5 ┘                       │
                                                                          └─(P2 + N3)─► AimRefine ─► aim (heatmap, dx, dy)
RepHG(c, mid, n): y0=x; y_i = RepConv(y_{i-1}); out = Conv1x1(concat(y0..y_n)) + x
Head level: [cls: 1×1/3×3 → n_cls] [box: RepConv → 4] [kp: RepConv ×2 → 4K]   ─► один тензор (cls | box | kp | score | vis)
```

Выход уровня `(1, n_cls+4+4K, H, W)` = 54 канала для K=12: `cls(2) | box(4) | смещения(24) | оценка точки(12) | видимость(12)`.
Раскладка лежит в метаданных ONNX (`apollo`), декодеру не нужен код обучения.

| вариант | параметры | GMACs @640 (развёрнутая) | где |
|---|---|---|---|
| `n` | 0.88 M | **1.17** (backbone 0.48, neck 0.28, head 0.34, aim 0.07) | кандидат на < 4 мс |
| `s` | 1.87 M | **2.33** (backbone 1.29, neck 0.38, head 0.50, aim 0.16) | точность / пропускная способность |
| `m` | 4.44 M | 5.74 | потолок точности, **не** для цели по латентности |

MACs считает `dataopen detector bench --ckpt ...` по развёрнутой модели (один 3×3 вместо трёх веток).

### Голова: декодирование (хост)

`anchor = ((gx+.5)s, (gy+.5)s)`; бокс `l,t,r,b = exp(clip(raw,-6,6))·s`; точка `= anchor + raw·4·s`; оценки `sigmoid`.
Выбор: порог `conf`, затем top-20 по оценке, **без NMS** (по умолчанию). Если в метриках появятся дубли, `nms_iou` включается
флагом, а частота дублей видна в `evaluate`.
Уточнение прицельной точки: soft-argmax heatmap (stride 4) в окне вокруг регрессированной точки, смещение внутри ячейки
добавляется; результат принимается, только если остался в `refine_radius_px` (6 px) от регрессии.

## 3. Результат для потребителя: `KeypointArray`

Ваш набросок почти годится; исправления: `visibility` в 8 бит не вмещает 12 точек (теперь `uint16 vis_mask`); целые
пиксели дают до 0.7 px ошибки прицеливания на голове в несколько пикселей (теперь Q12.4, шаг 1/16 px); добавлены `magic`,
`version`, `schema_hash` (потребитель отказывается принимать массив от другого набора точек), параметры letterbox и размер
кадра (координаты возвращаются в камеру без внешних данных), `track_id` (0 = нет трекинга). Структуры упакованы (`#pragma pack(1)`),
размеры проверяются `_Static_assert` и тестом: `ApolloDetection` 74 байта, `KeypointArray` 1516 байт.

```c
typedef struct {
    uint8_t  class_id;        // 0 player_ct, 1 player_t
    uint8_t  confidence;      // score*255
    uint16_t vis_mask;        // бит i = точка i видима (v==2)
    uint16_t track_id;
    uint16_t bbox[4];         // x,y,w,h  Q12.4, пиксели входа модели
    int16_t  keypoints[12][2];// Q12.4
    uint8_t  kp_confidence[12];
} ApolloDetection;
typedef struct { uint32_t magic; uint16_t version; uint8_t n_keypoints, max_detections; uint32_t schema_hash;
    uint32_t frame_id, timestamp_us; uint8_t detection_count, avg_scene_brightness, flags, reserved;
    float letterbox_scale; int16_t pad_x, pad_y; uint16_t src_w, src_h; ApolloDetection detections[20]; } KeypointArray;
```
Заголовок генерируется (`dataopen detector header`), лежит в `src/dataopen/detector/csrc/apollo_detection.h`. Постобработка на C99
(`csrc/apollo_post.c`: декод, top-K, уточнение, упаковка, без зависимостей) собирается тестом и сверяется с Python-эталоном на
случайных тензорах (допуск 1 LSB). Вызовы рантайма NPU и потоки остаются за интегратором.

## 4. Обучение

Полный путь по пунктам вашего ТЗ:

1. **Сбор кадров** (карты, скины, свет, дым, вспышки, дальность, позы): `dataopen collect --schema shooter12 --adaptive ...`,
   несколько шардов, `dataopen merge`. Замкнутый цикл отбраковывает мусор и генерирует сложные случаи вокруг головы (docs/QUALITY.md,
   docs/SCHEMAS.md). Сцены делятся на train/val **целиком**, поэтому val содержит неизвестные модели карты/скины/освещение.
2. **Аннотация COCO Keypoints и двойная проверка.** Разметку даёт движок (проекция скелета + hull). Для независимой проверки:
   `dataopen agreement A.json B.json --thr 0.85` сравнивает два источника (например, боксы из instance-масок игры, второй проход
   разметки или детекции обученной модели) и печатает долю пар с IoU ≥ 0.85, несопоставленные и список кадров на ревью. **Источника
   масок в модах пока нет**: без него вторая проверка это либо внешний аннотатор, либо модель.
3. **Обучение** (`dataopen detector train --config configs/apollo_s.toml --data runs/a runs/b --out runs/apollo_s`).

| Параметр | Значение |
|---|---|
| Оптимизатор | AdamW, lr 2e-3 (batch 64), wd 0.025 (не на BN/bias), betas (0.9, 0.999) |
| Расписание | линейный прогрев 3 эпохи, CosineAnnealing до 1% lr |
| Прочее | EMA 0.9998, AMP (bf16/fp16), clip 10, 120 эпох, последние 15 без mosaic |
| Аугментации | mosaic 1.0, mixup 0.1, HSV (0.015/0.7/0.4), flip 0.5 (точки меняются по `flip_idx` схемы), scale ±50%, translate 10%, поворот ±4°, фотометрические дым/вспышка 0.2 |
| Выборка | вес кадра = `weight` из замкнутого цикла (`hard_sampling = 1`), сложные кадры видны чаще |
| Потери | cls (IoU-aware BCE) + CIoU + OKS-потеря с весами схемы (голова ×3) + калиброванная оценка точки (цель: её собственная схожесть) + видимость + heatmap/смещение прицельной точки; обе головы |
| Назначение | task-aligned (α 0.5, β 6), top-10 для one-to-many, top-1 для one-to-one |

Валидация идёт по EMA-модели в развёрнутом виде. Лучшая контрольная точка: `0.5·map50 + 0.5·kp_ap_oks50`.

**Метрики** (`dataopen detector eval --require`): `map50` (class-aware AP@IoU 0.5), `kp_ap_oks50` (COCO keypoint AP при OKS 0.5 с
весами схемы), `aim_hit_rate`, `class_acc`, `recall_oks50`, и всё то же отдельно для **сложного подмножества** (кадры с весом ≥ 1.5).
Цель ТЗ: `map50 > 0.90`, `kp_ap_oks50 > 0.85`; `--require` возвращает код 1, если не достигнуто. Я бы дополнительно требовал цель
на сложном подмножестве: модель, хорошая в среднем, ломается именно там.

## 5. Экспорт

`dataopen detector export --ckpt best.pt --out m.onnx --input float|uint8`: берётся EMA, все RepConv/BN сворачиваются, 1/255 уходит
в первый conv, one-to-many голова удаляется, веса слияния запекаются в скалярные Mul. Экспорт **сверяется с PyTorch** (расхождение
больше 2e-3 останавливает выдачу файла). Тест проверяет, что в графе только `Conv, Relu, Add, Mul, Concat, Resize(nearest),
Constant, Transpose, Cast`, а IR-версия ≤ 8 (читают и ONNX Runtime, и rknn-toolkit2).

## 6. INT8: план

**Схема:** веса int8 per-channel (симметрично), активации 8 бит per-tensor (асимметрично), ≥ 500 калибровочных кадров.

1. **Калибровочный набор** (`dataopen detector calib --data runs/a runs/b --out calib --n 500`): кадры расслоены по вердикту,
   времени суток, дальности, дыму, вспышке, «прицельная точка пропущена базовой моделью»; квота идёт по кругу по слоям, поэтому редкие
   экстремальные условия представлены сильнее их частоты (процентильная калибровка не должна их обрезать); 40% `keep_hard`.
   Выход: PNG 640×640, `dataset.txt` (формат RKNN), `calibration_report.json` (покрытие слоёв).
2. **PTQ в ONNX Runtime** (`dataopen detector quantize --onnx m.onnx --calib calib --out m_int8.onnx`): QDQ, per-channel, процентильная
   калибровка. Печатает **косинусную близость по каждому выходу** FP32 против INT8, а `detector eval` оценивает INT8-модель теми же
   метриками. Это **заместитель** NPU (числа RKNN другие), но он ловит проблемы архитектуры на любой машине.
3. **RKNN:** `--rknn-script convert_rknn.py` пишет скрипт для rknn-toolkit2 (`asymmetric_quantized-8`, `quantized_method="channel"`,
   `mmse`, `accuracy_analysis`). Запускается на x86 Linux; **здесь не запускался**.
4. **Если точность падает больше допустимого** (порог: потеря kp_ap_oks50 > 2 п.п.): (a) `accuracy_analysis` / `compare_outputs`: какие
   выходы и слои теряют; (b) смешанная точность: последние 1×1 головы и aim-ветка в INT16/FP16 (`hybrid_quantization` в RKNN);
   (c) `mmse`/`kl_divergence`; (d) **QAT** (не реализовано): дообучение 10–20 эпох с фейк-квантованием на развёрнутой модели. Известный риск:
   RepVGG после слияния даёт каналы с большим разбросом весов; per-channel это в основном лечит.

Почему выходы удобны для INT8: box в лог-масштабе (диапазон ≈ [−2, 4]), смещения точек в единицах 4·stride (≈ ±1.5), логиты без
softmax; exp/sigmoid/top-k на хосте.

## 7. Бенчмарк и бюджет

`dataopen detector bench --ckpt best.pt --onnx m.onnx --rknn-script bench_rknn.py`: MACs по блокам, арифметическая оценка,
латентность FP32 на CPU хоста (**не NPU**), и скрипт для платы (`rknnlite`): латентность p50/p99 на всех трёх ядрах (core fusion) и
на одном ядре, пропускная способность тремя потоками по одному ядру, проверка порогов 4 мс и 240 FPS.

Арифметика (RK3588: 6 INT8 TOPS = 3 ядра по 2; 1 MAC = 2 операции; плотные свёртки реально дают 20–50% пика):

| вариант | время только вычислений, все ядра, 35% | одно ядро, 35% | 3 ядра по разным кадрам |
|---|---|---|---|
| n (1.17 GMACs) | 1.1 мс | 3.3 мс | ≈ 900 FPS |
| s (2.33 GMACs) | 2.2 мс | 6.7 мс | ≈ 450 FPS |

Это **нижняя граница**: не учтены память, преобразования layout и передача кадра. Цели противоречивы по смыслу: 240 FPS **на
3 ядрах по разным кадрам** это ≈ 12.5 мс на кадр на ядро, а < 4 мс на кадр требует либо модели, которая на одном ядре укладывается в 4 мс
(это `n`), либо core fusion (даёт ускорение меньше 3× и только на RK3588). Реально измерьте оба режима.

## 8. Команды

```bash
dataopen collect --game <игра> --schema shooter12 --frames 200000 --quality-model baseline.onnx --adaptive --out runs/a
dataopen verify runs/a; dataopen agreement runs/a/annotations/coco_train.json second_source.json
dataopen detector train --config configs/apollo_s.toml --data runs/a --out runs/apollo_s
dataopen detector eval --model runs/apollo_s/best.pt --data runs/a --require
dataopen detector export --ckpt runs/apollo_s/best.pt --out apollo_s.onnx --input float
dataopen detector calib --data runs/a --out calib --n 500
dataopen detector quantize --onnx apollo_s.onnx --calib calib --out apollo_s_int8.onnx --rknn-script convert_rknn.py
dataopen detector eval --model apollo_s_int8.onnx --data runs/a --require
dataopen detector bench --ckpt runs/apollo_s/best.pt --rknn-script bench_rknn.py       # затем на плате: python bench_rknn.py
# обученная модель как базовая в замкнутом цикле (раунд 2):
dataopen collect --game <игра> --schema shooter12 --quality-model apollo_s.onnx --quality-format apollo --adaptive --out runs/b
```

Замкнутый цикл и детектор образуют петлю: сильная модель-судья отбирает всё более трудные кадры, на которых обучается следующая
версия (см. раздел про смещение отбора в docs/QUALITY.md: по-прежнему выбрасывается только по признакам, не зависящим от модели).

## 9. Ограничения

- Всё про плату (конвертация, INT8-точность, латентность, FPS) требует вашего железа; здесь есть арифметика, скрипты и заместитель.
- На реальных данных не обучалось. Прогноз «map50 > 0.90» без данных дать нельзя; мелкие головы на дальности и дым/вспышки
  первыми покажут предел.
- Heatmap-ветка (stride 4) даёт заметную долю MACs у `s` (0.16 GMACs); отключается `aim_refine = false` ценой точности прицела.
- Классы определяются командой из меты сущности (`team`); для реальных игр мод должен её отдавать.
- Сжатие изображений и реальная оптика камеры не моделируются; фотометрические аугментации дыма/вспышки примитивны.
