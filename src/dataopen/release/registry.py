"""What the system is today and what 'ready for a pilot' means (docs/V1.md): modules with an honest status, the end-to-end scenarios, the
Definition of Done, the risks and the order of work. It is DATA, and `check.py` holds it against the repository: every file a module names
exists, every test a claim cites exists, a claim about hardware or people can not be marked met by editing a flag.

Status of a module (the strongest thing that can honestly be said about it today):

    implemented    complete code whose correctness a host can decide by itself: formats, protocols, crypto, state machines, algorithms with
                   reference vectors, C cores that are bit-exact with their references, tooling. Real-world fit is a separate question
    simulated      complete code that 'works' only against SIMULATORS of hardware, radio, USB hosts or people (or synthetic data)
    architecture   described (documents, tables, data models), with no running code of the thing itself
    deferred       consciously not part of v1

Nothing in this repository has run on a board, a real USB host, a real phone radio, a real monitor or a real person."""

from __future__ import annotations

from dataclasses import dataclass

IMPLEMENTED, SIMULATED, ARCHITECTURE, DEFERRED = "implemented", "simulated", "architecture", "deferred"
STATUSES = (IMPLEMENTED, SIMULATED, ARCHITECTURE, DEFERRED)
STATUS_TITLES = {IMPLEMENTED: "Implemented: код готов, корректность решается на хосте", SIMULATED: "Simulated: работает только против симуляторов (железа, радио, людей, данных)",
                 ARCHITECTURE: "Architecture-only: описано, кода нет", DEFERRED: "Deferred: сознательно вне v1"}


def t(path: str, *names: str) -> tuple:
    """Evidence refs: `t('test_x', 'a', 'b')` -> ('tests/test_x.py::a', 'tests/test_x.py::b'); no names -> the whole file."""
    p = path if "/" in path else f"tests/{path}.py"
    return tuple(f"{p}::{n}" for n in names) if names else (p,)


def js(path: str, *titles: str) -> tuple:
    p = f"pwa/tests/{path}"
    return tuple(f"{p}::{n}" for n in titles) if titles else (p,)


@dataclass(frozen=True)
class Module:
    id: str
    area: str
    name: str
    status: str
    what: str
    paths: tuple = ()
    evidence: tuple = ()
    gaps: tuple = ()
    doc: str = ""


AREAS = ("data", "perception", "assist", "bridge", "control", "security", "updates", "factory", "hardware", "integration")
AREA_TITLES = {"data": "Данные и обучение", "perception": "Зрение (видео и детекторы)", "assist": "Помощь (профиль, ASC, дрожь)", "bridge": "HID-мост",
               "control": "Управление (шлюз, телефон, слоты)", "security": "Личные данные и ключи", "updates": "Обновления", "factory": "Завод и сервис",
               "hardware": "Аппаратура и измерения", "integration": "Сборка в одну систему"}

MODULES = (
    # ---- data and training: the origin of the repository; not on the device
    Module("dataset-pipeline", "data", "Конвейер синтетических датасетов из игр (ядро, адаптеры, QA, замкнутый цикл качества)", IMPLEMENTED,
           "Генерация и проверка датасетов Human Pose; нужна только для обучения детекторов", ("src/dataopen/core", "src/dataopen/adapters", "src/dataopen/quality"),
           t("test_pipeline") + t("test_qa") + t("test_quality_integration") + t("test_protocol_conformance"),
           ("C#-плагины игр разбираются, но не собирались и не запускались", "на реальных играх ничего не проверялось", "в устройство не входит"), "docs/ARCHITECTURE.md"),
    Module("pose-detector", "data", "Детектор позы людей (ApolloNet-Pose): обучение, INT8/QAT, экспорт", IMPLEMENTED,
           "Исследовательский и аналитический инструмент; к цепочке помощи не подключён и не подключается", ("src/dataopen/detector",),
           t("test_detector_host") + t("test_detector_torch"),
           ("RKNN-конвертация и скорость на NPU не проверены", "mAP на реальных данных неизвестен", "в устройство не входит"), "docs/DETECTOR.md"),
    Module("runtime", "data", "Runtime инференса (цикл, политики кадров, каналы результатов, кольцо кадров)", IMPLEMENTED,
           "Нужен pose-детектору; кольцо кадров и политики общие с видеопутём", ("src/dataopen/runtime",), t("test_runtime"),
           ("`RknnLiteBackend` написан по документации, не запускался", "dma-buf без железа не проверен", "жёстких реального времени нет (обычный Linux)"), "docs/RUNTIME.md"),
    # ---- perception for the assistive path
    Module("video-prep", "perception", "Видеопуть: тайминги, правила EDID, подготовка кадра (C99), геометрия, доставка в FrameSource", IMPLEMENTED,
           "Потоковая подготовка 640×640 с точностью до 2 LSB к эталону; время появления строк; правила passthrough как модель", ("src/dataopen/video",),
           t("test_video"), ("всё на синтетических кадрах: ни одного реального сигнала HDMI/DP", "HDCP-флаг `hdcp_active` не существует (некому его выставить)"), "docs/VIDEO.md"),
    Module("video-hardware", "perception", "Захват и passthrough HDMI/DP на железе (чипы, ретаймер, демон V4L2/RGA/ISP, dma-buf)", ARCHITECTURE,
           "Выбор схем и демон захвата под SoC", (), (), ("номера чипов — кандидаты по памяти, требуют проверки по даташитам", "целостность сигнала, HDCP, мониторы не проверялись"),
           "docs/VIDEO.md"),
    Module("ui-detector", "perception", "Детектор элементов интерфейса и указателя: синтетика, обучение, трекер, сцена для ASC и моста, `UiService`", SIMULATED,
           "Единственный источник объектов для помощи; отдельный от pose-детектора мир, граница проверяется кодом", ("src/dataopen/ui",),
           t("test_ui") + t("test_ui_torch"),
           ("обучен только на синтетике: ни одного размеченного реального скриншота", "NPU, INT8 и RKNN не запускались",
            "в сборке по умолчанию очередь кадров FIFO: худший случай сцены выходит за TTL (docs/LATENCY.md); `latest_only` выключен по умолчанию"), "docs/UIDET.md"),
    # ---- assistance
    Module("bioprofile", "assist", "BioProfile: профиль моторики, 98 байт, движок эпизодов, усталость, хранение", SIMULATED,
           "Измеряет время реакции, перелёты, слежение, тремор из мыши и целей", ("src/dataopen/bioprofile",), t("test_bioprofile"),
           ("точность известна только против симулятора с известной истиной", "записанной моторики людей нет"), "docs/BIOPROFILE.md"),
    Module("asc-tremor", "assist", "ASC (вязкая зона у объекта с guard) и подавитель дрожи: float, fixed-point и C99, бит-в-бит", IMPLEMENTED,
           "Цепочка только вычитает: |out| ≤ |in|, знак тот же, из нуля нуль; такт ~44-49 нс на хосте", ("src/dataopen/assist",),
           t("test_assist") + t("test_tremor"),
           ("польза для людей не проверена (петля «человек в контуре» — симуляция)", "время такта на Cortex-M не измерено",
            "подавитель сдвигает намеренное движение: типично 20 мс, в начале броска до 46 мс, снимает ~13% чистого пути (синтетический пользователь)"), "docs/ASSIST.md"),
    # ---- the bridge
    Module("hid-bridge-core", "bridge", "HID-мост: ядро на C (дескрипторы, правка X/Y, образ устройства, fail-safe автомат, SPI-протокол, защиты границы доверия)", SIMULATED,
           "Прозрачный прокси мыши; мягкий PASSTHRU без разрыва, жёсткий bypass, watchdog, Panic; фаззинг под ASan/UBSan, сборка под Cortex-M7",
           ("src/dataopen/bridge",), t("test_bridge") + t("test_bridge_failsafe"),
           ("USB-поведение настоящих мышей, драйверов ПК и ПО производителя мышей не проверено", "время перенумерации ПК принято 300 мс (допущение)",
            "время такта на M7 не измерено"), "docs/BRIDGE.md"),
    Module("hid-bridge-board", "bridge", "HID-мост на плате: порт (TinyUSB host+device, SPI DMA, GPIO, таймер), реле/коммутатор, безопасный домен, RC-петля Panic", ARCHITECTURE,
           "Схема и порт под MCU", (), (), ("целостность HS-сигнала через коммутатор, ток, ESD не проверены", "реле и RC-петля существуют как таблица истинности и модель"),
           "docs/HARDWARE.md"),
    Module("failsafe-logic", "bridge", "Логика аппаратного fail-safe (таблица истинности, времена, согласование с моделью моста)", SIMULATED,
           "Проверена по всем сочетаниям входов и против `HwModel`", ("src/dataopen/hw/failsafe.py",), t("test_hw_failsafe"),
           ("`RELEASE_MS = 8` и `WDG_MS = 100` — допущения, на железе не измерены",), "docs/HARDWARE.md"),
    # ---- control
    Module("gateway-ctllink", "control", "Шлюз: CtlLink v1, манифест органов управления, сила и дрожь, проба «оставить / вернуть», STOP, калибровка (интерфейс)", IMPLEMENTED,
           "Единственное, с чем говорит телефон; проверен против настоящего C-ядра моста", ("src/dataopen/ctl/gateway.py", "src/dataopen/ctl/protocol.py", "src/dataopen/ctl/manifest.py"),
           t("test_ctl_gateway") + t("test_ctl_protocol") + t("test_ctl_manifest"), ("часы и SPI — симуляция", "сцену детектора шлюз не отправляет (см. `module-daemon`)"), "docs/PWA.md"),
    Module("pwa-client", "control", "Пульт на телефоне (PWA): сессия, WebSocket и BLE-транспорт, оболочка безопасности, манифест-рендерер, офлайн", SIMULATED,
           "Страница без криптографии, без аккаунтов; 360×800; ru/en", ("pwa",), js("e2e.test.mjs") + js("live-slots.test.mjs") + js("live-packages.test.mjs") + t("test_pwa"),
           ("настоящий Web Bluetooth, MTU, фон/сворачивание не проверены (подставной BLE)", "iPhone не поддерживается", "TalkBack / Switch Access с людьми не проверены"), "docs/PWA.md"),
    Module("ble-gatt-server", "control", "GATT-сервер на BlueZ и окно сопряжения по кнопке", ARCHITECTURE,
           "Сторона устройства для телефона", (), (), ("в шлюзе есть состояние окна сопряжения, сторона BlueZ не написана",), "docs/PWA.md"),
    Module("slots-panel", "control", "Четыре слота, панель (две кнопки, четыре светодиода), удержания 3/10/20 с", SIMULATED,
           "Профиль, уровни, раскладка, имя, ключ у каждого слота; смена не включает помощь", ("src/dataopen/ctl/slots.py", "src/dataopen/ctl/panel.py"),
           t("test_ctl_slots") + t("test_ctl_panel"), ("кнопки и светодиоды — модель; пороги 10 и 20 с надо подбирать с людьми",), "docs/SLOTS.md"),
    # ---- security
    Module("identity-seal-vault", "security", "Идентичность устройства, защищённый файл DOBS v2, шифрование данных на диске, криптографическое стирание", IMPLEMENTED,
           "Ключи владельца = Device Public ID; файл для одного устройства; повтор, подмена, физическое подтверждение", ("src/dataopen/ctl/identity.py", "src/dataopen/ctl/seal.py", "src/dataopen/ctl/vault.py"),
           t("test_ctl_identity") + t("test_ctl_seal") + t("test_ctl_vault"), ("ключи в файле, secure element не подключён", "внешней проверки криптографии нет"), "docs/SECURITY.md"),
    # ---- updates
    Module("update-channel-a", "updates", "Канал A: образ DOFW, два банка, проба, самоподтверждение, анти-откат, поток FW_*", SIMULATED,
           "Правила и протокол обновления системы", ("src/dataopen/ctl/firmware.py",), t("test_ctl_firmware") + t("test_ctl_update"),
           ("загрузчика нет: банки это файлы", "аппаратного счётчика анти-отката нет",
            "поднятый порог анти-отката делает золотой образ устаревшим: восстановление и полный возврат откажут, пока образ не обновят (правило выпуска: `release/policy.py`)"), "docs/UPDATES.md"),
    Module("bootloader", "updates", "Загрузчик SoM и MCU: выбор банка, счётчик загрузок, secure boot, золотой образ", ARCHITECTURE,
           "То, что исполняет правила канала A на плате", (), (), ("нет кода: правила закреплены тестами `SlotManager`",), "docs/UPDATES.md"),
    Module("update-channel-b", "updates", "Канал B: пакет DOPK, приём по частям, проверка весов модели, применение с roll-forward, возврат", IMPLEMENTED,
           "Профиль, уровни, раскладка, имя, веса лёгкой модели; для одного устройства", ("src/dataopen/updates",),
           t("test_updates_package") + t("test_updates_gateway") + t("test_updates_cli") + js("live-packages.test.mjs"),
           ("присланная модель не исполняется в песочнице: проверяется структура графа", "каталога официальных моделей нет"), "docs/UPDATES.md"),
    Module("residency", "updates", "Резидентность: профиль и веса модели живут только на устройстве (закрытая поверхность, единая дверь исхода, канарейки, шифртекст бесполезен после сброса)", IMPLEMENTED,
           "Нет выгрузки профиля и модели ни в каком виде; телефон видит только метаданные и три грубых признака готовности; поддержка только удаляет",
           ("src/dataopen/ctl/residency.py", "src/dataopen/ctl/residency_probe.py"),
           t("test_residency") + t("test_acceptance", "test_a_pilot_day_in_one_directory") + js("residency.test.mjs"),
           ("**root на модуле читает ключ диска из файла: нужен secure element**", "ключи вне чипа в симуляции", "независимой проверки не было",
            "цена правила: нет резервной копии профиля и переноса на запасное устройство"), "docs/RESIDENCY.md"),
    Module("custom-models", "updates", "Своя модель интерфейса: человек (или клиника) приносит ONNX пакетом от своего отправителя; контракт ABI v1, статическая проверка, кнопка на каждую модель, снятие и возврат", IMPLEMENTED,
           "Владелец готовит модель под нужные программы, проверяет её теми же правилами на компьютере и грузит через PWA без производителя и без магазина; на устройстве она резидентна",
           ("src/dataopen/updates/models.py", "src/dataopen/updates/sender.py", "src/dataopen/ui/resident.py"),
           t("test_models_custom") + t("test_models_torch") + js("models.test.mjs") + js("live-packages.test.mjs"),
           ("**проверка не доказывает, что модель не находит людей или игроков: метки, формы и операторы, не поведение**", "демона модуля нет: модель сохраняется и грузится в детектор, но в видеопуть не подключена",
            "ONNX на CPU через ORT; NPU (RKNN) не сделан", "разбор ONNX идёт в процессе шлюза", "пределы 4 GMAC и 2000 узлов это допущения",
            "ключ отправителя в файле (симуляция)", "решение по политике (U-3) не принято"), "docs/MODELS.md"),
    Module("scene-health", "perception", "Здоровье пути сцены: безопасные умолчания v1, автомат деградации с гистерезисом, отзыв сцены, бюджет инференса модели", IMPLEMENTED,
           "Либо помощь по сцене вовремя, либо предсказуемо без неё (не мигает); тяжёлая модель не ломает бюджет молча",
           ("src/dataopen/ui/policy.py", "src/dataopen/ui/health.py", "src/dataopen/ui/scene.py", "src/dataopen/ui/service.py"),
           t("test_scene_policy") + t("test_latency_budget") + t("test_latency_measure"),
           ("только виртуальное время и симуляция: SPI, DRDY, NPU, RGA, планировщик не измерены", "демона модуля нет: `scene_service` никто не вызывает в боевой сборке (D-1)",
            "скала TTL в ядре моста не менялась", "оценка стоимости модели линейная по одной сети"), "docs/LATENCY.md"),
    # ---- factory and service
    Module("provisioning", "factory", "Заводской провижининг: серийник, DAK, аттестация, станция из 9 шагов, стенд на настоящем ядре моста, цепочка доверия", SIMULATED,
           "Каждое устройство получает свой ключ и серийник без ручных действий", ("src/dataopen/provisioning/records.py", "src/dataopen/provisioning/station.py", "src/dataopen/provisioning/device.py"),
           t("test_prov_records") + t("test_prov_provision") + t("test_prov_cli"), ("OTP, secure element, HSM, база серийников и стенд — модели", "видео и BLE на стенде проверяются параметрически"), "docs/PROVISIONING.md"),
    Module("recovery-reset", "factory", "Восстановление (лестница слоёв 0-4) и четыре уровня сброса, полный возврат по токену", SIMULATED,
           "Что остаётся рабочим в худшем случае и как вернуть устройство без пайки", ("src/dataopen/provisioning/recovery.py", "src/dataopen/provisioning/service.py"),
           t("test_recovery") + t("test_prov_levels"), ("система восстановления, маскром, сервисный порт — правила и модель", "утилиты поддержки для маскрома нет"), "docs/PROVISIONING.md"),
    Module("factory-hardware", "factory", "Заводское железо: стенд-фикстура, HSM, программатор OTP/SE, принтер наклеек", ARCHITECTURE,
           "То, что исполняет станцию в реальности", (), (), ("нет", ), "docs/PROVISIONING.md"),
    # ---- hardware and measurements
    Module("hardware-spec", "hardware", "Аппаратная спецификация: четыре SKU через DNP, питание, тепло, корпус, BOM, индикация (данные и тесты)", ARCHITECTURE,
           "Решения и расчёты в виде проверяемых данных; платы, корпуса и замеров нет", ("src/dataopen/hw",),
           t("test_hw_spec") + t("test_hw_power") + t("test_hw_indication") + t("test_hw_cli"), ("PCB, корпус, BOM по реальным ценам, сертификация — нет", "тепло — оценка по формуле"), "docs/HARDWARE.md"),
    Module("latency-budget", "hardware", "Бюджет задержки: стадии с пометкой источника цифры, измерения в симуляции, разбор снятых данных стенда", SIMULATED,
           "Три группы (ввод, сцена, политика); методика и точки измерения на железе", ("src/dataopen/latency",), t("test_latency_budget") + t("test_latency_measure") + t("test_latency_cli"),
           ("ни одной цифры с железа", "джиттера планировщика и USB-хоста в симуляции нет"), "docs/LATENCY.md"),
    # ---- integration
    Module("acceptance", "integration", "Приёмочные сценарии на одном устройстве: завод → OOBE → калибровка → работа → слоты → пакет → OTA → Panic → восстановление → сбросы", SIMULATED,
           "Единственное место, где всё работает вместе на одном каталоге", ("tests/test_acceptance.py", "tests/acc_helpers.py"), t("test_acceptance"), ("всё — симуляция", ), "docs/V1.md"),
    Module("module-daemon", "integration", "Демон модуля: видео → UiService → сцена → SPI вместе со шлюзом (один мастер SPI), надзор за процессами, запуск", ARCHITECTURE,
           "Корень сборки на SoM: сегодня шлюз и сцена детектора работают в разных тестах и не делят один SPI-мастер", (), (),
           ("в приёмке сцена подаётся в ядро моста «вторым отправителем»; на устройстве мастер один и нужен мультиплексор кадров", "задержка ожидания SPI не определена"), "docs/LATENCY.md"),
    Module("live-calibration", "integration", "Живая калибровка: телеметрия моста + сцена → BioProfile на устройстве", ARCHITECTURE,
           "Источник профиля для реального человека", (), (), ("интерфейс `Learner` и симулятор есть, живой реализации нет: на железе профиль сегодня взять негде (кроме файла от другого устройства)",), "docs/PWA.md"),
    # ---- deferred
    Module("ios-softap", "control", "iPhone: Wi-Fi SoftAP вместо Web Bluetooth", DEFERRED, "В браузерах на iOS нет Web Bluetooth", (), (), ("только Android Chrome",), "docs/PWA.md"),
    Module("official-models", "updates", "Каталог официальных моделей, подписанных производителем и раздаваемых открыто", DEFERRED, "Другая политика доверия: не привязан к устройству", (), (), (), "docs/UPDATES.md"),
    Module("model-sandbox", "updates", "Исполнение присланной модели в песочнице с лимитами", DEFERRED, "Нужна, если в v1 разрешены модели от других людей", (), (), ("решение для v1: либо песочница, либо веса только от отправителей, которым человек доверяет кнопкой",), "docs/UPDATES.md"),
    Module("sound-profile", "hardware", "Звуковой профиль (громкость, отключение)", DEFERRED, "Зуммер описан, профиль в прошивке не реализован", (), (), (), "docs/HARDWARE.md"),
    Module("two-mice-delta-export", "updates", "Две мыши одновременно, дельта-обновления, RK3566-вариант Base", DEFERRED, "Не нужно для пилота", (), (), (), "docs/UPDATES.md"),
)
BY_ID = {m.id: m for m in MODULES}


# ---------------------------------------------------------------------------------------------------------------- scenarios
@dataclass(frozen=True)
class Scenario:
    id: str
    name: str
    pre: tuple
    steps: tuple
    expect: tuple
    covered: tuple          # evidence refs that exist today
    covered_text: str
    unverified: tuple
    acceptance: str         # the function in tests/test_acceptance.py that walks it


SCENARIOS = (
    Scenario("ACC-01", "Первый запуск (OOBE)",
             ("устройство прошло завод (состояние SHIPPED, серийник на наклейке)", "мышь в порту, кабель к ПК, питание от отдельного USB-C", "телефона рядом нет"),
             ("включить питание", "дождаться, пока ПК увидит мышь", "подключить приложение", "сверить серийник с наклейкой и цепочку производителя", "двигать мышь",
              "попробовать вынести или стереть без руки на устройстве", "открыть окно сопряжения удержанием CONFIRM 3-10 с"),
             ("мост в PASSTHRU: каждый отчёт мыши дошёл неизменным, ничего не потеряно", "помощь выключена, слотов с профилем нет, доверенных нет",
              "серийник в приложении = наклейке, карточка владельца подписана DAK, аттестация производителя проверяется", "жизненный цикл IN_FIELD",
              "без кнопки на устройстве нельзя ни вынести профиль, ни стереть, ни забыть телефоны", "окно сопряжения закрывается само через 120 с"),
             t("test_acceptance", "test_acc01_first_run") + t("test_prov_provision", "test_one_board_from_blank_to_shipped") + t("test_ctl_gateway", "test_a_fresh_device_does_not_assist_until_asked") +
             t("test_ctl_panel", "test_holding_confirm_three_seconds_opens_the_pairing_window_and_it_closes_by_itself") + t("test_prov_levels", "test_the_gateway_shows_the_serial_and_a_card_with_the_manufacturers_chain"),
             "завод, шлюз, мост, панель и PWA-протокол на одном каталоге",
             ("настоящий Web Bluetooth и сопряжение на телефоне (GATT-сервера нет)", "первое включение на настоящей плате и настоящем ПК", "понятность первого запуска для человека",
              "печатная инструкция и наклейка"), "s01_first_run"),
    Scenario("ACC-02", "Калибровка профиля и первое включение помощи",
             ("OOBE пройден", "в слот приходят наблюдения: движения мыши и цели"),
             ("запустить калибровку из приложения", "попытаться включить помощь во время калибровки", "работать, пока «профиль собран» не дойдёт до 100 %", "остановить калибровку",
              "включить помощь (проба 20 с)", "нажать «Оставить»; второй раз не нажимать"),
             ("во время калибровки помощь выключена и включить её нельзя, указатель не тронут", "после остановки в слоте есть профиль, помощь ещё выключена",
              "включение — проба, которая сама откатывается; «Оставить» делает её постоянной", "параметры принял мост без отказов"),
             t("test_acceptance", "test_acc02_calibration_to_help") + t("test_ctl_gateway", "test_calibration_learns_a_profile_while_assistance_is_held_off", "test_raising_help_is_a_trial_that_undoes_itself") +
             t("test_bioprofile", "test_reaction_time_and_flick_metrics_are_recovered"),
             "шлюз + BioProfile на симулированном человеке + настоящее C-ядро",
             ("**живой калибровки нет**: нет `Learner` на данных моста и сцены, на железе профиль взять негде", "точность профиля на записанных людях", "достаточна ли длительность калибровки (5 минут сима)",
              "утомление человека во время калибровки"), "s02_calibration"),
    Scenario("ACC-03", "Обычная работа помощи",
             ("есть профиль, помощь включена и оставлена", "сцена приходит от детектора (в приёмке — вторым отправителем)"),
             ("двигать мышь без целей", "потянуться к цели со сценой, возраст которой 20 мс", "отключить телефон", "нажать STOP в приложении", "перезапустить модуль", "включить помощь снова"),
             ("на каждом отчёте |out| ≤ |in|, знак тот же, из нуля нуль", "подход к цели притормаживается (K<0.8), указатель не подталкивается", "потеря телефона ничего не меняет",
              "STOP: следующий отчёт байт в байт как у мыши, помощь сама не вернётся после перезапуска"),
             t("test_acceptance", "test_acc03_everyday_work") + t("test_assist", "test_invariants_hold_on_random_input", "test_the_c_core_is_bit_identical_to_the_python_fixed_point_model") +
             t("test_tremor", "test_never_amplifies_never_flips_never_creates_motion_and_removes_at_most_the_tremor") + t("test_ctl_gateway", "test_stop_is_immediate_persisted_and_byte_exact", "test_losing_the_phone_changes_nothing_about_the_device") +
             t("test_ui", "test_frames_to_service_to_scene_frames_to_the_real_bridge_core_brake_the_approach") + t("test_latency_measure", "test_a_scene_keeps_its_help_until_the_core_ttl_and_then_gives_none"),
             "цепочка шлюз → мост (C) → отчёты ПК; сцена через ядро; STOP и потеря связи",
             ("**детектор → сцена → мост через общий SPI-мастер с шлюзом** (в приёмке второй отправитель)", "поведение на реальных экранах и реальных программах", "польза и неудобство для людей: типичное отставание фильтра дрожи 20 мс, до 46 мс в начале броска",
              "реальные мыши, 125 Гц и 8 кГц, мыши с ПО производителя", "задержки на железе (docs/LATENCY.md)"), "s03_everyday_work"),
    Scenario("ACC-04", "Переключение слотов",
             ("два контекста: у каждого свой профиль, уровни, имя", "помощь включена"),
             ("кнопка СЛОТ ходит только по занятым слотам", "новое содержимое слота (не «в работе») включить как пробу и нажать «Вернуть»", "выбрать слот с телефона и «Оставить»",
              "сравнить уровни слотов", "очистить слот с кнопкой"),
             ("смена слота не включает и не выключает помощь", "светодиод, телефон и кнопка согласны", "пробный слот откатывается к прежнему сразу", "параметры моста — свои у каждого слота, зажим держится",
              "очистка слота стирает только его, остальные не тронуты"),
             t("test_acceptance", "test_acc04_slots") + t("test_ctl_slots", "test_switching_a_slot_never_turns_assistance_on", "test_a_slot_kept_in_work_switches_at_once_but_a_new_one_is_a_trial_that_undoes_to_the_old_slot",
                                                         "test_the_safety_clamp_holds_in_every_slot", "test_clearing_a_slot_makes_its_old_ciphertext_dead_and_leaves_the_others_alone") +
             t("test_ctl_panel", "test_the_slot_button_goes_round_the_slots_that_are_in_use") + js("live-slots.test.mjs", "the SLOT button on the device moves the slot and the phone hears about it"),
             "шлюз + мост + панель + PWA-клиент против настоящих шлюза и ядра",
             ("кнопки и светодиоды на плате (дребезг, подсветка на солнце, читаемость)", "удобство для людей с тремором (короткое нажатие 30 мс - 2 с)", "переключение по окну программы — вне v1"), "s04_slots"),
    Scenario("ACC-05", "Загрузка пакета (профиль, уровни, раскладка, модель)",
             ("помощь работает", "файл `.dopk` от отправителя, которому человек доверяет (устройство пакетов само не делает)"),
             ("загрузить файл на странице «Ещё»", "подделанный файл и файл для другого устройства", "оборвать связь посередине и продолжить", "применить без кнопки, затем с кнопкой",
              "применить модель, потом ещё одну, вернуть прежнюю", "послать системный образ как пакет и пакет как системный образ"),
             ("поддельный и чужой пакеты отвергаются первыми кусками, ничего не сохранено", "передача идёт, пока человек работает: ни одной потери отчётов", "ничего не применяется само; кнопка нужна новому отправителю и весам не от владельца",
              "повтор отвергается, прежнюю модель можно вернуть", "каналы не подменяют друг друга", "модель в слоте зашифрована ключом слота и наружу не выходит (в ответах только короткий id)"),
             t("test_acceptance", "test_acc05_packages") + t("test_updates_package") + t("test_updates_gateway") + js("live-packages.test.mjs", "a forged, a misaddressed and a tampered package are refused"),
             "формат, поток, шлюз, слоты, мост и клиент",
             ("присланная модель не исполняется в песочнице (в v1 нужно решение: песочница или веса только от доверенных отправителей)", "качество присланной модели", "большие файлы по реальному BLE (скорость, обрывы)", "понятность сообщений об отказах для людей"), "s05_packages"),
    Scenario("ACC-06", "Обновление системы (OTA)",
             ("помощь работает", "подписанный производителем образ для этого типа железа"),
             ("загрузить подделанные и чужие образы", "загрузить хороший образ при работающей помощи", "применить без кнопки, затем с кнопкой", "дождаться пробы и подтверждения", "попытаться вернуть старую версию файлом",
              "применить образ, который не докажет себя"),
             ("плохие образы не доходят до банка", "работающий банк и мышь не тронуты загрузкой; модуль перезапускается, мост носит мышь без потерь", "новая версия сама подтверждает себя при здоровом мосте, порог анти-отката растёт",
              "старый образ вернуть нельзя; данные человека не тронуты", "неподтверждённая версия заменяется прежней сама, мышь работает всё время"),
             t("test_acceptance", "test_acc06_system_update") + t("test_ctl_update") + t("test_ctl_firmware", "test_power_loss_during_the_write_leaves_the_running_firmware_booting", "test_every_single_bit_of_an_image_is_protected"),
             "правила банков, поток, шлюз с перезапуском, мост без потерь",
             ("**загрузчик, flash и secure boot на плате**", "настоящий обрыв питания посреди записи", "первое обновление уже выпущенной партии", "золотой образ и порог (правило выпуска) в реальном процессе сборки"), "s06_system_update"),
    Scenario("ACC-07", "Panic и bypass",
             ("помощь включена", "кнопка Panic и мышь доступны"),
             ("короткое нажатие Panic", "просить помощь с телефона во время защёлки", "удержать 2 с (переармирование)", "удержать 3 с (жёсткий bypass)", "зависание прошивки моста", "STOP без сеанса"),
             ("короткое: помощь выключается мгновенно, указатель не прерывается, перенумерации нет, отчёты байт в байт", "телефон защёлку не снимает и сообщает почему",
              "длинное: контакты возвращают настоящую мышь, управление не нужно прошивке", "при зависании реле отпускается в окне watchdog 100 мс", "STOP работает без HELLO и без манифеста"),
             t("test_acceptance", "test_acc07_panic_and_bypass") + t("test_bridge_failsafe") + t("test_hw_failsafe") + t("test_ctl_gateway", "test_the_phone_can_not_lift_a_panic_latch_and_says_why", "test_hardware_bypass_is_one_way_from_the_phone", "test_nothing_but_the_safety_actions_works_before_hello"),
             "автомат моста, модель реле и RC-петли, шлюз, телефон",
             ("**реле, RC-петля и watchdog на плате**: времена 100 мс, 3 с и 8 мс — допущения", "перенумерация на разных ПК и в BIOS (300 мс принято): при зависании мышь может пропасть на 0.4-2 с", "кнопка Panic в руках человека с тремором (форма, усилие)",
              "целостность HS-сигнала через коммутатор"), "s07_panic_and_bypass"),
    Scenario("ACC-08", "Восстановление после сбоя",
             ("работающее устройство в поле",), ("модуль умер целиком, мост жив", "модуль вернулся на то же хранилище", "повреждён второй банк прошивки", "все сценарии лестницы восстановления"),
             ("мост без модуля пропускает мышь без потерь и без помощи", "слоты, доверие и желание человека на месте, помощь возвращается сама", "повреждённый банк не принимается и ничего не ломает",
              "в каждом сценарии у мыши есть путь; все, кроме возврата изготовителю, заканчиваются рабочей системой"),
             t("test_acceptance", "test_acc08_recovery") + t("test_recovery") + t("test_bridge_failsafe", "test_stale_parameters_or_a_silent_module_drop_to_passthrough_and_recover_by_themselves"),
             "поведение моста без модуля + МОДЕЛЬ лестницы восстановления",
             ("**система восстановления, золотой образ, маскром и сервисный порт — модель, не код**", "утилиты поддержки нет", "учения с настоящим устройством без пайки", "время восстановления для человека без компьютера"), "s08_recovery"),
    Scenario("ACC-09", "Сбросы четырёх уровней",
             ("устройство в использовании: слоты, модель, ждущий пакет, доверенные, телефоны",),
             ("L1 очистить слот (кнопка)", "L2 стереть все профили (кнопка)", "L3 заводской сброс (кнопка)", "L4 полный возврат: токен производителя + рука + золотой образ", "после L3 файл, сделанный для старого владельца, не открывается"),
             ("L1 стирает слот и его модель, остальное остаётся; L2 стирает всё личное, владелец и телефоны остаются, отправители забыты (нужна кнопка)", "L3: новый владелец, телефоны забыты, файлы для старого номера мертвы, серийник и DAK те же",
              "L4: токен одноразовый и привязан к серийнику; устаревший золотой образ отказывает ДО траты токена; банки заново, порог не снижен, история в OTP", "никакой уровень не меняет серийник, DAK, аттестацию и порог"),
             t("test_acceptance", "test_acc09_resets") + t("test_prov_levels") + t("test_recovery", "test_the_full_return_reinstalls_the_images_and_clears_the_person_but_not_the_device", "test_the_token_the_hand_and_the_challenge_are_all_required"),
             "матрица сверена с настоящим шлюзом; L4 на каталоге устройства",
             ("secure element и OTP: серийник и DAK на настоящем кристалле", "физическое стирание flash (здесь стирание = замена ключа)", "утилита поддержки и процесс выдачи токенов в реальной организации"), "s09_resets"),
)
SCENARIO_DAY = "test_a_pilot_day_in_one_directory"


# ---------------------------------------------------------------------------------------------------------------- Definition of Done
HARDWARE, HUMAN, SOFTWARE, PROCESS = "hardware", "human", "software", "process"
KINDS = (HARDWARE, HUMAN, SOFTWARE, PROCESS)
KIND_TITLES = {HARDWARE: "нужно железо", HUMAN: "нужны люди", SOFTWARE: "нужен код", PROCESS: "нужен процесс или решение"}


@dataclass(frozen=True)
class Dod:
    id: str
    group: str
    criterion: str
    kind: str
    met: bool = False
    evidence: tuple = ()
    record: str = ""        # a file that records the sign-off for a hardware / human criterion (without it such a criterion can not be met)
    note: str = ""


V1_SCOPE = ("Изделие: DO-1 Base (HDMI, USB-A, одна мышь), пилот под наблюдением, до нескольких человек, Android + Chrome.",
            "Входит: прозрачный мост мыши (DIRECT / SPLIT / SLOW), Panic и аппаратный bypass, подавитель дрожи, ASC при наличии сцены, пульт на телефоне, четыре слота, профиль и уровни пакетом, системное OTA, заводской провижининг, восстановление и четыре уровня сброса.",
            "Включено только при решении: модели от других людей (см. DoD U-3), сцена от UI-детектора на реальных экранах (D-A4).",
            "Не входит: iPhone, модели из каталога, дельта-обновления, две мыши, Pro-функции (Wi-Fi, звук).")

DOD = (
    # ---- safety first
    Dod("S-1", "Безопасность", "Аппаратный bypass возвращает настоящую мышь при мёртвой прошивке, снятом питании, оборванном кабеле кнопки, на реальной плате и реле", HARDWARE),
    Dod("S-2", "Безопасность", "Времена Panic, watchdog и отпускания реле измерены на плате и укладываются в допущения (100 мс, ≈3 с, ≤ 8-10 мс)", HARDWARE),
    Dod("S-3", "Безопасность", "Время перенумерации измерено на Windows, macOS, Linux и в BIOS/UEFI; худший случай «мышь пропала при зависании» записан в инструкцию", HARDWARE),
    Dod("S-4", "Безопасность", "Прозрачность проверена на не менее чем 10 разных мышах (в том числе игровых с ПО производителя, 125 Гц, 1 кГц, 8 кГц, трекболы)", HARDWARE),
    Dod("S-5", "Безопасность", "Логика fail-safe, инварианты «только вычитание» и фаззинг ядра моста: проверены симуляцией и проходят", SOFTWARE, True,
        t("test_bridge_failsafe") + t("test_hw_failsafe") + t("test_assist", "test_invariants_hold_on_random_input"), note="симуляция, не плата"),
    Dod("S-6", "Безопасность", "Профиль и веса модели не покидают устройство: в протоколе, манифесте, клиенте и хранилище нет пути выгрузки; известные байты (канарейки) ни разу не вышли; после сброса шифртекст бесполезен", SOFTWARE, True,
        t("test_residency") + t("test_acceptance", "test_a_pilot_day_in_one_directory") + js("residency.test.mjs"), note="симуляция; не защищает от root на модуле, пока ключи в файле (S-7)"),
    Dod("S-7", "Безопасность", "Ключи диска, слотов и владельца в secure element: root на модуле и снятая flash-память не дают ни профиля, ни весов", HARDWARE),
    # ---- assistance
    Dod("A-1", "Помощь", "Калибровка на устройстве из живых данных (мост + сцена или скриптовая цель) создаёт профиль без файла из другого места", SOFTWARE),
    Dod("A-2", "Помощь", "С несколькими людьми под наблюдением помощь не мешает: протокол пилота, согласие, возможность выключить, журнал только на устройстве", HUMAN),
    Dod("A-3", "Помощь", "Отставание подавителя дрожи (типично 20 мс, до 46 мс в начале броска, ~13% пути снимается) принято людьми или параметры подобраны", HUMAN),
    Dod("A-4", "Помощь", "UI-детектор проверен на размеченных реальных экранах целевых программ (порог precision/recall задан заранее), либо сцена в пилоте ограничена известными программами", HUMAN),
    Dod("A-5", "Помощь", "Помощь ASC в обычной работе не нарушает инварианты (|out| ≤ |in|, из нуля нуль) в цепочке «завод → день работы»", SOFTWARE, True,
        t("test_acceptance", "test_a_pilot_day_in_one_directory", "test_acc03_everyday_work"), note="симуляция"),
    Dod("A-6", "Помощь", "Своя модель, принесённая владельцем пакетом, исполняется детектором на SoM в бюджете задержки (путь ONNX → RKNN/NPU или CPU), качество измерено на экранах владельца", HARDWARE),
    # ---- the device software as one system
    Dod("D-1", "Сборка", "Демон модуля: видео → UI-детектор → сцена → SPI вместе со шлюзом (один мастер SPI), запуск и надзор процессов; приёмочная цепочка проходит на нём", SOFTWARE),
    Dod("D-2", "Сборка", "GATT-сервер BlueZ и окно сопряжения по кнопке работают с реальными телефонами Android (Chrome), потеря связи и фон не ломают сеанс", HARDWARE),
    Dod("D-3", "Сборка", "PWA проверено с TalkBack и Switch Access людьми, которым оно предназначено (цели ≥ 64 px, контраст, порядок)", HUMAN),
    Dod("D-4", "Сборка", "Приёмочные сценарии ACC-01…09 и «день пилота» проходят на одном устройстве в симуляции", SOFTWARE, True,
        t("test_acceptance"), note="симуляция"),
    Dod("D-5", "Сборка", "Безопасные значения по умолчанию в боевой сборке: `UiService(latest_only=True, max_age_ms)`; сцена в пределах TTL с запасом", SOFTWARE, True,
        t("test_scene_policy") + t("test_latency_budget") + t("test_latency_measure"),
        note="умолчания закреплены в коде сервиса и выведены из бюджета; боевой сборки и демона нет (D-1), числа NPU это цель"),
    # ---- updates
    Dod("U-1", "Обновления", "Загрузчик с двумя банками, счётчиком загрузок, secure boot и аппаратным порогом анти-отката на SoM и MCU; обрыв питания при записи проверен", HARDWARE),
    Dod("U-2", "Обновления", "Правило выпуска «порог после обновления не выше версии золотого образа» проверяется функцией и тестом", SOFTWARE, True,
        t("test_release", "test_a_release_that_would_outrun_the_golden_image_is_refused"), note="функция есть; встроить в реальный конвейер выпуска — часть процесса"),
    Dod("U-3", "Обновления", "Решение по моделям от других людей: песочница с лимитами, либо в v1 веса принимаются только от доверенных отправителей (кнопка на устройстве каждый раз)", PROCESS,
        note="политика предложена и реализована в симуляции (docs/MODELS.md: ABI, отправитель владельца, кнопка на каждую модель); решение принимает владелец продукта"),
    Dod("U-4", "Обновления", "Первая реальная партия обновлена OTA с платы на плату и возвращена: порядок выпуска, подпись в HSM, золотой образ синхронизирован", HARDWARE),
    # ---- security and factory
    Dod("P-1", "Завод и ключи", "Ключи владельца и DAK в secure element, OTP и защита отладки заблокированы на реальных кристаллах; серийники не повторяются", HARDWARE),
    Dod("P-2", "Завод и ключи", "Ключ производителя в HSM, церемония подписи, отзыв и ротация описаны и отработаны", PROCESS),
    Dod("P-3", "Завод и ключи", "Заводской стенд с реальной оснасткой и базой серийников прошёл не менее 5 плат подряд, брак уходит в карантин", HARDWARE),
    Dod("P-4", "Завод и ключи", "Независимый обзор безопасности протокола, форматов и хранения (в том числе канала B и BLE-сопряжения), замечания закрыты", PROCESS),
    Dod("P-5", "Завод и ключи", "Заводская цепочка (провижининг, цепочка доверия, уровни сброса) проходит тесты в симуляции", SOFTWARE, True,
        t("test_prov_provision") + t("test_prov_levels") + t("test_prov_records"), note="симуляция"),
    # ---- quality, measurements, field
    Dod("Q-1", "Качество", "Автотесты Python и Node зелёные в CI на каждом коммите", SOFTWARE, True, t("test_acceptance") + t("test_pwa"), note="см. CI репозитория"),
    Dod("Q-2", "Качество", "Задержки измерены на железе: ввод ≤ 1 мс при 1 кГц и сцена с запасом до TTL на целевом SoC (docs/LATENCY.md, раздел 6.2)", HARDWARE),
    Dod("Q-3", "Качество", "Тепло: пассивное охлаждение в закрытом корпусе при 25-35 °C в пределах для мыши и видео", HARDWARE),
    Dod("Q-4", "Качество", "Путь сертификации выбран и пройден до пилота: EMC, электробезопасность, USB, медицинская классификация или заявление «не медицинское изделие»", PROCESS),
    Dod("Q-5", "Качество", "Поддержка: утилита восстановления, выдача токенов, процедура RMA и замена отработаны на реальном устройстве (учения без пайки)", HARDWARE),
    Dod("Q-6", "Качество", "Пилот: ответственный, журнал инцидентов, процедура остановки, данные только на устройстве, инструкция в доступной форме", PROCESS),
)
DOD_BY_ID = {d.id: d for d in DOD}


# ---------------------------------------------------------------------------------------------------------------- risks and the order of work
@dataclass(frozen=True)
class Risk:
    id: str
    level: str      # high | medium | low
    title: str
    why: str
    cure: str


RISKS = (
    Risk("R-1", "high", "Ничего не проверено на железе", "Ни одно реле, USB-соединение, время, ток, сигнал HDMI/DP, радио и ни одна мышь не видели этого кода. Все допущения симулятора (перенумерация 300 мс, реле 8 мс, watchdog 100 мс) могут оказаться неверными",
         "стенд на плате (шаг 3): S-1…S-4 и Q-2 — до всего остального"),
    Risk("R-2", "high", "Нет сборки в одну систему", "Шлюз и сцена детектора работают в разных тестах; на устройстве SPI-мастер один. Живой калибровки нет — настоящий профиль взять негде. Первая сборка на железе найдёт то, чего симулятор не покажет",
         "корень сборки в симуляции (шаг 2): демон + мультиплексор кадров + живой Learner, приёмочная цепочка поверх него"),
    Risk("R-3", "high", "Польза для людей не доказана, есть цена", "Всё про пользу — симуляция человека. Подавитель дрожи сдвигает намеренное движение (типично 20 мс, до 46 мс) и снимает ~13% пути; помощь может раздражать. Пороги кнопок 10 и 20 с для людей с тремором не проверены",
         "наблюдаемый пилот с протоколом (шаг 7); возможность выключить, быстрая настройка уровней"),
    Risk("R-4", "high", "UI-детектор не видел ни одного реального экрана", "Обучен на синтетике; на реальных программах качество неизвестно, ложные цели тормозят указатель в неожиданных местах. Вывод на CPU 11-16 мс; прежняя очередь FIFO давала худший случай сцены 118 мс при TTL 100 мс; умолчания v1 (`latest_only`, `max_age_ms`) возвращают его под TTL (≈ 60 мс, при использовании ≈ 76 мс), а при выходе из бюджета помощь по сцене выключается, а не мигает",
         "реальные скриншоты и разметка (шаг 6), NPU; в пилоте — ограничить сцену известными программами"),
    Risk("R-5", "high", "Обновление и восстановление без загрузчика", "Правила закреплены тестами, но flash, secure boot, золотой образ, маскром и утилита поддержки — только модель. Подъём порога анти-отката делает золотой образ устаревшим: самовосстановление и полный возврат откажут, пока образ не заменят",
         "загрузчик на плате (шаг 5); правило выпуска (U-2) в конвейере; учения по восстановлению"),
    Risk("R-6", "high", "Безопасность не рассматривал никто, кроме авторов", "Ключи в файлах, secure element не подключён, ключ производителя в тестовом виде, BLE-сопряжение «Just Works» с окном по кнопке не проверено, модель от другого человека не исполняется в песочнице",
         "secure element и HSM (шаг 5), независимый обзор и пентест (P-4), решение по моделям (U-3)"),
    Risk("R-7", "medium", "Зависание прошивки моста оставляет мышь мёртвой на время перенумерации", "Окно watchdog 100 мс плюс перенумерация ПК (принято 300 мс, возможно 0.5-2 с): при зависании человек может остаться без указателя на секунды",
         "измерить на ПК (S-3), короткое окно watchdog, инструкция"),
    Risk("R-8", "medium", "Телефон как единственный пульт", "Android Chrome; iOS нет; настоящий Web Bluetooth не проверен. Смягчение: безопасные действия (Panic, bypass, слоты, подтверждение) есть на самом устройстве",
         "GATT-сервер и тесты на реальных телефонах (D-2); при необходимости SoftAP"),
    Risk("R-9", "medium", "Регуляторика и ответственность", "Ассистивное устройство для людей с нарушениями: EMC, электробезопасность, USB-соответствие, классификация (медицинское или нет), защита данных, совместимость с играми и античитом",
         "выбрать путь до пилота (Q-4)"),
    Risk("R-10", "medium", "Тесты и код написаны теми же людьми с теми же допущениями", "Общие допущения проходят все проверки одновременно. Независимых тестов и тестов «железо в контуре» нет",
         "стенд с железом в контуре; независимый обзор"),
    Risk("R-11", "low", "Одна мышь, один компьютер, один язык интерфейса", "Две мыши не поддерживаются; интерфейс ru/en", "сознательно вне v1"),
    Risk("R-12", "high", "Резидентность держится на ключах в файле и на отсутствии кода", "Пока ключ диска лежит в файле, root на модуле расшифрует профиль и веса; закрытая поверхность защищает от телефона, поддержки и ошибок протокола, а не от того, у кого железо. Цена правила: профиль нельзя выгрузить, значит нет резервной копии и переноса на запасное устройство",
         "secure element и закрытая отладка (S-7, шаг 5); решение о политике замены устройства (калибровка заново)"),
    Risk("R-13", "high", "Статическая проверка чужой модели не проверяет поведение", "Метки, формы выходов, список операторов, привязка к устройству, доверенный отправитель и кнопка на каждую модель это барьеры. Модель с честными метками всё равно может находить на экране людей или игроков; сама цепочка помощи только вычитает, но цель выбирают веса. Плюс: забывание отправителя (`trust.clear`) забывает и его номера пакетов, старый перехваченный пакет можно предъявить снова (нужны две кнопки)",
         "указатель на экране о том, что видит помощь; решение U-3; песочница и исполнение под лимитами (шаг сборки); при необходимости подпись не только отправителя, но и клиники (общий список)"),
    Risk("R-14", "medium", "Бюджет задержки держится на цифрах, которых на плате нет", "Все цифры пути сцены получены в виртуальном времени на CPU-замере 11/16 мс и целях NPU 3/4 мс; SPI, DRDY, RGA, джиттер планировщика и часы не измерены. Оценка стоимости пользовательской модели по числу MAC линейна и построена по одной сети: тяжёлая модель может оказаться медленнее оценки",
         "стенд L1-L5 по LATENCY.md; рантайм-страж p95 инференса уже выключает помощь по сцене; замер на RK3588 (шаг 4, A-6)"),
)


@dataclass(frozen=True)
class Step:
    n: int
    title: str
    what: str
    exit: tuple         # DoD ids that must be met to leave the step
    depends: tuple = ()


ORDER = (
    Step(1, "Заморозить объём v1 и включить безопасные умолчания", "Принять V1_SCOPE; умолчания сцены уже в коде (`ui/policy.py`): демон вызывает `scene_service(...)`; решение по моделям от других людей (песочница или только владелец); правило выпуска «порог ≤ золотой образ» в конвейер", ("D-5", "U-2", "U-3")),
    Step(2, "Собрать систему в симуляции", "Демон модуля: один мастер SPI для шлюза и сцены, видео → UI-детектор → сцена; живой Learner (телеметрия моста + цели из UI-детектора или скриптовой калибровочной программы); приёмочная цепочка поверх демона", ("D-1", "A-1", "D-4"), (1,)),
    Step(3, "Стенд моста на плате", "Порт моста на dev-плату с реле/коммутатором, RC-петлёй Panic и watchdog; настоящая мышь и ПК; измерить перенумерацию, путь ввода, тайминги bypass; 10 мышей, три ОС", ("S-1", "S-2", "S-3", "S-4"), (1,)),
    Step(4, "Модуль: BlueZ, захват, NPU", "GATT-сервер и окно сопряжения; HDMI-захват и подготовка кадра на RK3588; UI-детектор на NPU; измерить задержки по docs/LATENCY.md", ("D-2", "Q-2", "A-6"), (2, 3)),
    Step(5, "Загрузчик, ключи, завод", "Двухбанковый загрузчик и secure boot на SoM и MCU; secure element, OTP и блокировка отладки; станция с реальной оснасткой на 5 платах; HSM и церемония; учения восстановления с маскромом", ("U-1", "U-4", "P-1", "P-2", "P-3", "Q-5", "S-7"), (3, 4)),
    Step(6, "Реальные экраны для UI-детектора", "Собрать и разметить скриншоты целевых программ, дообучить, задать порог precision/recall; либо ограничить сцену в пилоте известными программами", ("A-4",), (2,)),
    Step(7, "Люди", "Протокол и согласие; наблюдаемые сессии с несколькими людьми; подбор уровней, порогов кнопок, компромисс отставания фильтра дрожи; проверка PWA с TalkBack и Switch Access", ("A-2", "A-3", "D-3"), (4, 6)),
    Step(8, "Безопасность и сертификация", "Независимый обзор и пентест; путь сертификации и классификация; тепло в корпусе", ("P-4", "Q-4", "Q-3"), (5,)),
    Step(9, "Пилот", "Ограниченная партия, ответственный, журнал инцидентов, процедура остановки, RMA; данные только на устройстве", ("Q-6",), (7, 8)),
)

NEVER = ("Адаптер «враги из онлайн-игры» и подключение pose-детектора к цепочке помощи: помощь в прицеливании по другим игрокам — это не accessibility, границы записаны в docs/ASSIST.md и docs/UIDET.md",
         "Сокрытие устройства от ПО и античита: мост не прячется, сервисы вправе не принимать оборудование, меняющее ввод",
         "Обход HDCP и сохранение копии экрана: кадры живут в оперативной памяти и перезаписываются")
