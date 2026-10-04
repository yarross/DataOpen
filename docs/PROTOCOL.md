# Протокол core ↔ game mod (v1)

Для авторов модов. Реализации: `adapters/lua/runtime/dataopen_rpc.lua` (Lua), `adapters/unity/DataOpen.UnitySdk` (C#),
`src/dataopen/adapters/mock/server.py` (Python, эталон). Любая новая реализация обязана пройти `dataopen doctor`.

## Транспорт: файловый «почтовый ящик»

Одна папка, два файла. Никаких сокетов, листинга каталогов и переименований на стороне игры.

```
core  → <dir>/req.json   {"v":1,"id":N,"method":"...","params":{...}}     пишет атомарно (tmp + rename)
game  → <dir>/res.json   {"id":N,"result":{...}}  |  {"id":N,"error":{"message":"...","type":"..."}}
```

- Мод обрабатывает запрос, если его `id` **отличается** от последнего обработанного. Одновременно один запрос.
- Ядро игнорирует ответ с чужим `id` и **повторяет чтение, если `res.json` оборван** (мод не обязан писать атомарно).
- Мод игнорирует `req.json`, который не парсится (ядро ещё пишет).
- Если `v` ≠ 1 — ответить `error` с типом `ProtocolError`.
- Папка должна быть доступна и игре, и ядру. GMod: `garrysmod/data/dataopen/`. BepInEx: `BepInEx/config/dataopen/`.
- Ядро запускается с таймаутами (`call_timeout_s`, `capture_timeout_s`); тишина ⇒ ошибка с путём ящика.

## Соглашения

- Единицы — **метры** (мод переводит дюймы/сантиметры сам; `doctor` ловит сантиметры).
- Камера передаётся **позой**: `pos` и мировые единичные векторы **экранных** осей `forward`, `right`, `up` + вертикальный
  `fov_v_deg` (или `fov_h_deg`). Проекция в ядре: `x=(p-pos)·right`, `y=-(p-pos)·up`, `z=(p-pos)·forward`,
  `u=W/2+f·x/z`, `v=H/2+f·y/z`. Руки координат и оси движка не важны: мод просто отдаёт базис.
- Пиксели непрерывные, начало в левом верхнем углу левого верхнего пикселя.
- Порядок ключевых точек — как в `schema.keypoints` из `hello`; позиции плоским списком `[x0,y0,z0,x1,...]`.

## Методы

| Метод | Параметры | Результат |
|---|---|---|
| `hello` | `protocol`, `schema{name,keypoints}`, `bone_map`, `options`, `image{width,height}` | `protocol`, `game`, `engine`, `game_version`, `mod_version`, `capabilities[]`, `image{width,height}`, `schema_errors[]`, `parameter_space` |
| `begin_scene` | `scene` (`environment`, `actors[]`, `area`, `seed`, ...) | `handles[{entity_id,rig_id,meta}]` (пусто в observe-режиме) |
| `capture_frame` | `frame_id`, `frame{camera, actor_frame[], kind}`, `width`, `height`, `active`, `image_mode` | снимок кадра (ниже) |
| `release` | — | `{}` разморозка (режим `host`) |
| `commit` | `frame_token`, `dest` | `{}` или `{"staged": "<путь относительно ящика>"}` |
| `discard` | `frame_token` | `{}` |
| `end_scene` | — | `{}` |
| `selftest` | — | `{"checks":[{name, ok, detail, hint?, data?}]}` |
| `health` | — | `{ok, frames, ...}` |
| `shutdown` | — | `{}` |
| `peek` | `frame_token`, `max_side?`, `shm?{name, capacity}` | `{transport:"shm", width, height, format:"rgb24"}` или `{transport:"file", staged, width, height}` |

`hello` — мод (пере)инициализирует состояние. `capabilities`: `probes`, `engine_visibility`, `hull_points`,
`image_engine`, `deterministic_step`, `image_peek`, `image_shm`. `image_engine` ⇒ мод сам пишет картинку по `commit`; иначе кадр снимает ядро
(`host`-режим: мод замораживает игру до `release`).

`parameter_space` — что игра умеет рандомизировать (ядро выбирает значения, мод применяет). Типы параметров:
`{"type":"uniform","lo","hi"}`, `loguniform`, `categorical` (`choices`, `weights?`), `constant` (`value`).
Группы: `environment` (дополняет значения ядра: `time_of_day`, `weather`, `cloud_cover`, `fog_density`, ...),
`actor` (на сцену), `actor_frame` (на кадр).

### Пиксели для проверки качества (`peek`)

`image_peek`: мод отдаёт пиксели ещё не закоммиченного кадра (для валидации в памяти, `docs/QUALITY.md`), **не расходуя**
кадр: `commit`/`discard` по-прежнему обязательны. `image_shm`: ядро создало сегмент общей памяти и сообщает его имя в `shm`;
мод пишет туда сырой RGB24, **строки сверху вниз**, и отвечает `transport:"shm"` (ядро читает без копии). Без общей памяти
(Lua) мод кладёт PNG в staged-файл и отвечает `transport:"file"`; ядро читает и удаляет файл. `max_side` — подсказка, её можно
игнорировать.

### Снимок кадра (`capture_frame`)

```jsonc
{
  "frame_token": "s000001_f0003",          // = frame_id запроса
  "tick": 12345,
  "camera": {"width":1280,"height":720,"pos":[x,y,z],"forward":[..],"right":[..],"up":[..],"fov_v_deg":60,"near":0.1},
  "entities": [{
    "entity_id": 7, "rig_id": "valvebiped",
    "skeleton_world": [x0,y0,z0, ...],     // K*3, метры, порядок схемы
    "joint_valid": [true, ...],            // false = кость не найдена
    "engine_visibility": [2,1,...],        // 2 виден, 1 перекрыт (raycast движка), по желанию
    "hull_points": [x,y,z, ...],           // границы меша, по желанию (точный bbox)
    "meta": {"forward":[..]}               // направление взгляда персонажа (для проверки лево/право), по желанию
  }],
  "probes": [{"world":[x,y,z],"screen":[u,v]}],  // screen — РОДНАЯ проекция движка; опустить, если вне экрана
  "warnings": ["..."]
}
```

**Инварианты:** кости, камера и пиксели принадлежат одному моменту движка. Кости читать там же, где рендерится кадр.
`probes` — 3–6 мировых точек и то, куда **сам движок** их проецирует; ядро сверяет со своей проекцией и при расхождении
отбраковывает кадр и подсказывает причину (перевёрнутая ось, FOV, единицы). Это защита от тихо испорченной разметки.

## Ошибки

Любая ошибка обработчика ⇒ `error` с понятным `message` (со стеком) и `type`; мод продолжает работать.
Ядро превращает их в `RemoteError`; повторяющиеся ошибки останавливают сессию с причиной.

## Staging (`commit`)

Мод в песочнице (GMod) не может писать в произвольный путь. Он кладёт файл в свою папку данных и отвечает
`{"staged":"staging/<token>.dat"}`; ядро перемещает файл в `dest`.

## Совместимость

`v` растёт только при несовместимых изменениях. Новые необязательные поля добавляются без смены версии: старый мод
их не пришлёт, ядро не требует.
