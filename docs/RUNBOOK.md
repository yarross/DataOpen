# Runbook: от установки до датасета

Статус проверки (честно): ядро, протокол, `doctor`, `verify`, Lua-рантайм и GMod-хост протестированы автоматически
(GMod — против заглушки API, не против настоящей игры). **C#-плагины и команды RCON на реальных Valheim/Rust не
запускались**: их проверка — это ваш первый запуск, и `dataopen doctor` для этого и сделан. Версии игр и модов меняются:
перед стартом зафиксируйте версию игры (Steam → Свойства → Бета-версии / отключить автообновление).

## 0. Установка ядра (один раз)

```bash
git clone <repo> && cd DataOpen
pip install -e '.[dev,capture]'
python -m pytest -q                       # должно быть зелёным
dataopen games                            # профили
dataopen serve-mock --mailbox /tmp/mbox & # опционально: пробный прогон без игры
dataopen doctor --game mock --mailbox /tmp/mbox
```

## 1. Порядок для любой игры (лесенка)

1. `dataopen install --game <gmod|valheim|rust> --dir "<папка игры>"`
2. Запустить игру (условия ниже), загрузить карту/мир.
3. Задать переменную папки игры (`GMOD_DIR`, `VALHEIM_DIR`, `RUST_DIR`) или передать `--mailbox`.
4. `dataopen doctor --game <игра>` → читать вывод сверху вниз: **первый FAIL — единственное, что нужно чинить**.
   Картинки с наложенным скелетом: `doctor_out/doctor/overlay_*.png`. Синие точки — левая сторона, красные — правая.
5. `dataopen collect --game <игра> --out runs/<имя> --frames 200 --provenance-note "<источник и лицензия ассетов>"`
   (перед сбором `collect` сам запускает быструю проверку и отказывается, если она красная).
6. `dataopen verify runs/<имя>` → `qa_report.md`; `dataopen preview runs/<имя>` → `preview.png`: **посмотрите глазами**.
7. Только после этого — большой сбор.

«Идеальный» датасет = `doctor` без FAIL + `verify` без errors + `preview` без визуальных ошибок + заполненная
`DATASET_CARD.json` (особенно `provenance`).

## 2. Garry's Mod (главный полигон)

Условия: запуск с параметром `-insecure`, одиночная игра или listen-сервер, карта с ровным открытым местом
(`map gm_flatgrass`). Окно игры в фокусе, **не открывать меню паузы** (в нём `Think` не работает).

```
dataopen install --game gmod --dir "C:\Program Files (x86)\Steam\steamapps\common\GarrysMod"
set GMOD_DIR=C:\Program Files (x86)\Steam\steamapps\common\GarrysMod
dataopen doctor --game gmod
```

В консоли игры: `dataopen_status` (мост жив ли), `dataopen_mark` (запомнить текущую точку; сцены идут по отмеченным
местам: больше разнообразия фонов). Модели персонажей берутся из всех установленных playermodel-аддонов.

Что проверить в первую очередь (мест, где я мог ошибиться в API, помечены `VERIFY` в `gmod_host.lua`):

| Симптом в doctor | Вероятная причина | Что делать |
|---|---|---|
| `mod:render_hook` FAIL | окно не в фокусе / пауза / `PostRender` не вызывается | вернуть фокус, выйти из меню |
| `projection` FAIL «focal length differs» | RenderView трактует FOV иначе | в профиле `mod_options.fov_mode = "horizontal"` или `"vertical"`, повторить |
| `projection` FAIL «Y axis is flipped» | перепутан знак `up`/`right` | написать мне вывод doctor |
| `mod:ground` FAIL | рядом нет ровной земли | встать на открытое место / `gm_flatgrass` |
| оверлей: скелет сдвинут/кривой на кадре | позы модели не обновились до чтения костей | увеличить `mod_options.settle_ticks` |
| `mod:data_folder_write` FAIL | нет прав на `garrysmod/data` | запуск от пользователя с правами |
| кадр чёрный/пустой | `render.Capture` не вернул данные | сменить формат/RT-флаги в `gmod_host.lua` (`VERIFY`) |
| `bone_mapping` FAIL | у модели нестандартный скелет | `[bones]` в профиле по списку `bones_found` |

Известные границы GMod: свет мира задаётся картой; вариативность освещения обеспечивают цветокоррекция, туман и смена
карт (запускайте сессии по разным картам и объединяйте через `dataopen merge`); персонажи — клиентские модели без
коллизий, поэтому перекрытие одним актёром другого определяется только эвристикой самоперекрытия, а не лучами.

## 3. Valheim

Условия: BepInEx 5 (Mono) для вашей версии Valheim установлен и один раз запускался; .NET SDK 6+; загрузитесь в мир.

```
dataopen install --game valheim --dir "C:\...\Valheim"      # собирает и кладёт DataOpen.Valheim.dll
set VALHEIM_DIR=C:\...\Valheim
dataopen doctor --game valheim
```

По умолчанию плагин **наблюдает** гуманоидов вокруг игрока (`Animator`). Чтобы он сам спавнил врагов, добавьте в профиль
`mod_options.spawn_enemies = true` (пассивный ИИ: `ai_passive`). Время суток/погода — через `EnvMan` по имени членов;
если имена в вашей версии другие, `doctor` покажет `valheim_envman`/`valheim_note`, а датасет можно собирать и без этого.

| Симптом | Причина | Что делать |
|---|---|---|
| `dotnet build failed` | другая версия Unity/BepInEx | прислать вывод и версию игры |
| `bone_mapping`: «no humanoid found» | рядом никого нет | `spawn_enemies = true` или подойти к NPC |
| `bone_mapping` FAIL | имена костей рига | `[bones]` по `bones_found` |
| `render` FAIL | SRP-пайплайн требует копии камеры | прислать вывод |
| `left_right` WARN/FAIL | у рига нестандартный `forward` | игнорируйте только если overlay корректен |

## 4. Rust (свой локальный сервер)

Только против **вашего** локального сервера. Клиент и сервер — два процесса:
- сервер: свой dedicated, RCON включён (`+rcon.web 1 +rcon.port 28016 +rcon.password <пароль>`; актуальные флаги и
  отключение EAC для собственного сервера — по официальной документации Facepunch);
- клиент: BepInEx 5 (Mono) + плагин; подключиться к своему серверу.

```
dataopen install --game rust --dir "C:\...\Rust"
set RUST_DIR=C:\...\Rust
set RUST_RCON_PASSWORD=<пароль>
dataopen doctor --game rust
```

Время суток и погода меняются на сервере командами из `[server] scene_begin` профиля (имена команд — `VERIFY`, правятся без
кода). Клиент только наблюдает и снимает. Игра тяжёлая и часто обновляется: после каждого обновления снова `doctor`.

## 4a. Контроль качества кадров (замкнутый цикл)

Подробно: [QUALITY.md](QUALITY.md). Три уровня, от простого к полному:

```bash
# 1) без модели: отсечь сломанные скелеты, пустые кадры и людей, неразличимых в пикселях (GPU не нужен)
dataopen collect --game gmod --out runs/a --frames 1000 --quality-static
# 2) с вашей моделью: проверьте её на одной картинке, потом включите
dataopen eval-image --model yolov8n-pose.onnx --image sample.png --keypoint-map coco17
dataopen collect --game gmod --out runs/a --frames 1000 --quality-model yolov8n-pose.onnx --quality-device cuda
# 3) плюс адаптивная рандомизация (учится, какие условия и внешности дают сложные кадры)
dataopen collect ... --quality-model yolov8n-pose.onnx --adaptive
```

После сбора смотрите `closed_loop_report.md` (что отброшено и почему, где модель страдает), `quality/rejects/` и
`quality/audit/` (глазами: не режем ли мы годные кадры). Много `quality_drop_invisible` на дневных кадрах значит, что порог
различимости слишком строгий (`[quality.features]`), а не что игра плоха. Модель не загрузилась: `device="tensorrt"` без
TensorRT тихо переключится на CUDA/CPU (предупреждение в логе); ставьте `onnxruntime-gpu` для GPU.

## 5. Масштабирование и эксплуатация

- **Несколько экземпляров игры**: у каждого свой ящик (`--mailbox`) и своя часть сцен:
  `dataopen collect ... --shard 0/4 --out runs/a0`, `--shard 1/4 --out runs/a1`, ...; затем
  `dataopen merge runs/all runs/a0 runs/a1 runs/a2 runs/a3` (отказывается при пересечении кадров).
- **Продолжить после сбоя**: та же команда с `--resume` (гранулярность — сцена, воспроизводимо по сидам).
- **Остановить аккуратно**: Ctrl+C (дописывает сцену и отчёты).
- **JPEG** вместо PNG: `--image-ext jpg` (нужен Pillow, `pip install -e '.[capture]'`), в разы меньше места.
- **Что смотреть в `report.json`**: `rejects` (много `unlabeled_person_present` ⇒ сузьте дистанцию камеры или увеличьте
  разрешение), `stage_seconds` (где время), `probe_failed` (должно быть 0).
- **Коды выхода**: 0 ок, 1 проверка не пройдена, 2 сбор недобран, 3 прерван из-за ошибки (калибровка/адаптер), 4 неверное использование.

## 6. Что делать, если ничего не помогает

Прислать: вывод `dataopen doctor`, версию игры и мода, `doctor_out/doctor/doctor_report.json`, `overlay_*.png`, лог игры
(`BepInEx/LogOutput.log` или `garrysmod/console.log`).
