# DataOpen

Синтетические датасеты Human Pose (COCO Keypoints / YOLO-Pose) из игровых движков.
**Universal Core** (Python) + **моды внутри игр** (Lua, C#). Игры: Garry's Mod, Valheim, Rust (свой сервер); любую другую
можно добавить за один профиль и один мод (`docs/MODDING.md`).

```bash
pip install -e '.[dev,capture]'
python -m pytest -q
make demo                      # весь конвейер без игры: doctor → collect → verify → preview

dataopen games                                          # профили
dataopen install --game gmod --dir "<папка игры>"       # поставить мод
dataopen doctor  --game gmod                            # проверить интеграцию (читайте первый FAIL)
dataopen collect --game gmod --out runs/a --frames 1000 --provenance-note "<источник и лицензия ассетов>"
dataopen verify  runs/a ; dataopen preview runs/a       # QA и контактный лист
dataopen requalify runs/a --policy strict.toml          # новые пороги качества на готовом датасете (docs/QUALITY.md)
dataopen detector train --config configs/apollo_s.toml --data runs/a --out runs/apollo_s   # детектор (pip install '.[train]')
dataopen runtime bench --backend mock --fps 240 --duration 5       # runtime инференса: задержка / FPS / дропы / очередь (docs/RUNTIME.md)
```

| Документ | О чём |
|---|---|
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | запуск на каждой игре, таблицы «симптом → что делать», масштабирование |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | устройство, решения, разметка и видимость, узкие места, риски |
| [docs/QUALITY.md](docs/QUALITY.md) | замкнутый цикл: проверка кадров моделью в памяти, адаптивная рандомизация, метаданные сложности |
| [docs/SCHEMAS.md](docs/SCHEMAS.md) | целевая схема keypoints (shooter12: 12 прицельных точек, 2 класса), как менять набор точек данными |
| [docs/DETECTOR.md](docs/DETECTOR.md) | детектор под Rockchip NPU: архитектура, обучение, экспорт, INT8, постобработка, бенчмарк (`dataopen detector ...`) |
| [docs/RUNTIME.md](docs/RUNTIME.md) | runtime на целевом SoC: главный цикл, политика «новый кадр раньше старого», IPC-канал `KeypointArray`, метрики, интеграция с `IModelEvaluator` (`dataopen runtime ...`) |
| [docs/BIOPROFILE.md](docs/BIOPROFILE.md) | BioProfile: персональный профиль биомеханики мыши (T_motor, флики, слежение, сценарии ошибок, усталость), компактная структура 98 байт, интерфейс чтения (`dataopen bioprofile ...`) |
| [docs/ASSIST.md](docs/ASSIST.md) | Adaptive Sensitivity Correction: вязкая зона вокруг объекта для людей с тремором/перелётами (K ∈ [0.1, 1], guard до начала движения, профиль из BioProfile, fixed-point + C ядро) и односторонний подавитель дрожи 3–16 Гц (`dataopen assist ...`) |
| [docs/BRIDGE.md](docs/BRIDGE.md) | Assistive HID Bridge: прозрачный USB-прокси мыши (все дескрипторы, кнопки и запросы как у настоящей мыши), правит только биты X/Y и только вычитанием; fail-safe (мягкий PASSTHRU без разрыва ввода, жёсткий аппаратный bypass, watchdog, Panic), SPI-протокол с модулем коррекции, симулятор установки (`dataopen bridge ...`) |
| [docs/VIDEO.md](docs/VIDEO.md) | Video-путь: HDMI/DP passthrough на монитор без добавленной задержки (EDID байт в байт) и параллельный захват; потоковая подготовка кадра (цвет, кроп, area-даунскейл, letterbox 640×640), геометрия экран ↔ вход детектора, доставка в runtime; тайминги и честная задержка (`dataopen video ...`) |
| [docs/UIDET.md](docs/UIDET.md) | Assistive UI-детектор элементов интерфейса и указателя (кнопки, значки, поля, меню, переключатели, вкладки, кнопки окна): отдельный мир от pose/people-детектора, синтетика + дообучение на реальных скриншотах, сцена для ASC и HID-моста (`dataopen ui ...`) |
| [docs/PROTOCOL.md](docs/PROTOCOL.md) | протокол core ↔ мод для авторов модов |
| [docs/MODDING.md](docs/MODDING.md) | как добавить игру |
| [docs/GAMES.md](docs/GAMES.md) | выбор игр и чек-лист для новой |

Честный статус: ядро, протокол, `doctor`/`verify`, Lua-рантайм и хост GMod (на заглушке API) покрыты тестами. C#-плагины
разбираются и проверяются на C# 7.3, но **не собирались и не запускались**, а на реальных играх ничего ещё не проверялось:
первый запуск `dataopen doctor` на игре и есть эта проверка.
