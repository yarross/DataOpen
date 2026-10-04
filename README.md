# DataOpen

Синтетические датасеты Human Pose (COCO Keypoints / YOLO-Pose) из игровых движков:
**Universal Core** (Python) + **Game Adapters** (Unity / Unreal).

Архитектура, интерфейсы, протокол кадра, узкие места и риски: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

```bash
pip install -e '.[dev]' && python -m pytest -q
python -m dataopen.cli --adapter mock --out out/run1 --frames 500
```
