# Как добавить игру

Решение за 5 минут: чек-лист из `docs/GAMES.md`. Дальше один из трёх путей.

## A. Игра со встроенным Lua (GMod) или Lua-модами (UE4SS, Cyber Engine Tweaks)

Рантайм `adapters/lua/runtime/dataopen_rpc.lua` уже реализует весь протокол (JSON, ящик, корутинные обработчики,
резолвер костей, пробы, таймауты, авторазморозка). Вам нужен **host** — таблица с привязками к движку.
Эталон: `adapters/lua/tests/mock_host.lua` (симуляция) и `adapters/gmod/addon/dataopen/lua/dataopen/gmod_host.lua`.

```lua
local R = dofile("dataopen_rpc.lua")
local host = { name = "mygame", engine = "...", unit_scale = 0.0254 }   -- метров на единицу движка
-- обязательные
host.fs = { read = function(name) ... end, write = function(name, data) ... end }   -- файлы ящика
host.now()                    -- секунды (реальное время)
host.parameter_space()        -- что рандомизируется (см. PROTOCOL.md)
host.bone_map(rig_id)         -- keypoint -> { {bone, weight}, ... } (готовые карты: R.maps.valvebiped, R.maps.ue_mannequin)
host.begin_scene(scene, rt)   -- применить окружение, создать/найти актёров; вернуть {{entity_id=, rig_id=, meta=}, ...}
host.set_active(handles, bool); host.apply_frame(frame, handles, rt)
host.place_camera(spec, handles, w, h, rt); host.read_camera(w, h) -> {pos, forward, right, up, fov_v_deg}
host.entities(rt) -> handles для этого кадра;  host.read_bones(handle) -> { name = {x=,y=,z=} }
host.project(point, w, h) -> {x=,y=}|nil   -- РОДНАЯ проекция движка (или host.project_batch для движков, где она
                                           -- доступна только в рендер-хуке)
host.freeze(); host.unfreeze()
-- по желанию
host.visibility(cam_pos, point, handle) -> bool   host.entity_extras(h) -> {forward=, hull_points=}
host.capture_image(token, w, h, rt)  host.commit_image(token, dest, rt) -> staged_name   host.discard_image(token)
host.sample_bones(rt) -> {rig_id=, bones=}   host.selftests(rt)   host.render_size()   host.init(options, rt, image)
```

Правила: код на Lua 5.1/LuaJIT-совместимом подмножестве (без `//`, битовых операторов, `goto`); внутри обработчика можно
ждать: `rt:wait_frames(n)`, `rt:wait_until(pred, timeout_s)`; вызывать `rt:poll()` раз за тик игры.
Положения — таблицы `{x=,y=,z=}` (поля с маленькой буквы); единицы движка — рантайм сам переводит в метры.

Тест без игры: скопируйте `tests/test_lua_runtime.py` / `tests/test_gmod_host.py`, замените мок/стаб на свой.
Рантайм проверяется на Lua 5.1, 5.4 и LuaJIT (`pip install lupa`).

**UE4SS** (Lua 5.4) и **Cyber Engine Tweaks** (LuaJIT): те же файлы; `host.fs` через `io.open` в папке мода, тик — из
`RegisterHook`/`LoopAsync` или `registerForEvent("onUpdate")`, кости через `GetBoneLocation`/слоты, проекция через
`ProjectWorldLocationToScreen`. Для игр, где камеру поставить нельзя, используйте режим «камера игрока + актёры вокруг»,
а где нельзя заспавнить — `population_mode = "observe"`.

## B. Unity-игра (Mono) через BepInEx

`adapters/unity/DataOpen.UnitySdk` не знает про конкретную игру: ищет `Animator`-гуманоидов в радиусе от якоря и
рендерит их свободной камерой. Новая игра = наследник `GenericBindings` (см. `DataOpen.Valheim`) только если нужно
управлять временем/погодой/спавном; члены игры достаются **по имени через `Reflect`**, поэтому плагин собирается без
ссылок на сборки игры, а неверное имя видно в `doctor`. Для Rust-подобных клиент-серверных игр окружение задаётся
сервером по RCON (`adapters/rcon.py`, шаблоны команд в профиле).

Новая игра: скопируйте `DataOpen.Rust` (папка, `.csproj`, `Plugin.cs`), поменяйте имена, добавьте профиль
`src/dataopen/game_profiles/<game>.toml` и запись в `installer._UNITY`. Исходники должны оставаться C# 7.3
(`tests/test_csharp_syntax.py` проверяет разбор и запрещает синтаксис новее).

## C. Нативный плагин (C++)

Только если A и B невозможны (Skyrim SKSE, REDengine RED4ext). Реализуйте тот же протокол: он простой (JSON + два файла).
Переиспользуйте `docs/PROTOCOL.md` и сверяйтесь с `dataopen doctor`.

## Профиль игры (`game_profiles/<id>.toml`)

Путь ящика, размер рендера, режим захвата, `mod_options`, `[bones]` (переопределение имён костей без перекомпиляции),
`[server]` (RCON), `[provenance]`. Образцы лежат рядом.
