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
```

| Документ | О чём |
|---|---|
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | запуск на каждой игре, таблицы «симптом → что делать», масштабирование |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | устройство, решения, разметка и видимость, узкие места, риски |
| [docs/PROTOCOL.md](docs/PROTOCOL.md) | протокол core ↔ мод для авторов модов |
| [docs/MODDING.md](docs/MODDING.md) | как добавить игру |
| [docs/GAMES.md](docs/GAMES.md) | выбор игр и чек-лист для новой |

Честный статус: ядро, протокол, `doctor`/`verify`, Lua-рантайм и хост GMod (на заглушке API) покрыты тестами. C#-плагины
разбираются и проверяются на C# 7.3, но **не собирались и не запускались**, а на реальных играх ничего ещё не проверялось:
первый запуск `dataopen doctor` на игре и есть эта проверка.
