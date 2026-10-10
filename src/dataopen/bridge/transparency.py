"""OS transparency / zero PC footprint (docs/TRANSPARENCY.md): what the PC and its software can and can not see of the bridge.

The requirement from accessibility and enterprise IT: the computer and any software on it must see an ORDINARY mouse. No driver of our own, no
VID/PID of a "helper", no background service, no feature reports from the adapter. Corporate policies, antivirus and old programs then have
nothing new to object to for a person who needs the help.

This module is the one place the properties, their evidence and the honest limits are written down; the tables in the document are generated
from it, and `tests/test_transparency.py` checks that every named test exists. It is a statement about the SIMULATION and the code: no real
mouse firmware, no real PC, OS or policy, and no USB analyzer was involved. The bridge is NOT built to hide from software that looks for it
(docs/BRIDGE.md): it is built so that nothing extra has to be installed, trusted or whitelisted."""

from __future__ import annotations

from dataclasses import dataclass

STATUS = {
    "sim": "доказано в симуляции (тест)",
    "code": "следует из устройства кода (проверяется чтением исходников и закреплённым списком функций)",
    "model": "модель в симуляции; электрическое свойство на плате не проверено",
    "open": "не сделано",
}


@dataclass(frozen=True)
class Prop:
    id: str
    requirement: str
    status: str
    how: str
    tests: tuple
    limit: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUS:
            raise ValueError(self.id)


PROPS = (
    Prop("T1", "На ПК не устанавливается никакой софт: ни драйвер, ни служба, ни приложение", "code",
         "У изделия нет ничего на стороне ПК: вход ПК это USB-порт устройства (клон мыши), пульт это телефон по BLE, видео снимается отдельным путём до монитора. "
         "Код видеозахвата и модуля коррекции работает на SoM внутри адаптера. Установщики модов для игр (`dataopen install`, `adapters/`) в изделие не входят",
         ("test_the_device_has_no_software_for_the_pc_in_its_scope", "test_nothing_in_the_core_can_describe_a_device_of_its_own"),
         "Если сам ПК потребует подтвердить новое USB-устройство (политика «разрешать только известные VID/PID»), то мышь с тем же VID/PID, что была, проходит, но политика по серийному номеру или по порту может отличать"),
    Prop("T2", "Система видит ту же мышь: Device Manager / lsusb показывают те же VID/PID, строки, серийный номер, интерфейсы, конечные точки", "sim",
         "Образ мыши (device, единственная configuration, HID class и report-дескрипторы) отдаётся ПК из кэша байт в байт; строки на всех языках, BOS, MS OS и всё остальное пересылается мыши вживую",
         ("test_the_pc_sees_the_same_mouse_in_every_state", "test_the_pc_only_ever_enumerates_the_identity_of_the_real_mouse"),
         "Проверено на пяти смоделированных мышах; настоящие мыши, их прошивки и драйверы вендоров не проверялись"),
    Prop("T3", "Полный клон дескрипторов и живой прокси control-запросов это обязательное свойство v1, не опция", "sim",
         "В конфигурации ядра нет поля, которое выключило бы клон или прокси; мост не включается (остаётся аппаратный bypass), если не может воспроизвести мышь точно; маршрутизация запросов проверена свойством на тысячах случайных запросов",
         ("test_cloning_and_the_live_proxy_are_not_options", "test_the_proxy_routing_follows_the_rule_for_any_request", "test_devices_the_proxy_can_not_mirror_exactly_are_never_engaged"),
         "Мышь, которую нельзя отразить точно (изохронные/bulk конечные точки, несколько конфигураций, альтернативные настройки, абсолютный указатель), мост не обслуживает: она работает напрямую и без помощи"),
    Prop("T4", "Правятся только относительные X/Y, и только уменьшением", "sim",
         "`hid_desc.c` пишет только биты X/Y; финальный зажим по осям на выходе: `|out| ≤ |in|`, знак тот же или ноль; ввод ноль даёт вывод ноль",
         ("test_only_xy_bits_change_and_only_downwards_for_any_report", "test_extreme_and_adversarial_inputs_keep_the_per_axis_invariants"),
         "Это поведение видно ПК по определению (указатель движется иначе, чем рука): прозрачность идентичности, а не движения"),
    Prop("T5", "Кнопки, колесо, Report ID, вендорные отчёты и все прочие интерфейсы передаются как есть", "sim",
         "Нетронутые интерфейсы копируются 1:1 (HID++-подобный канал, клавиатура); в отчёте движения меняются только X/Y; длина отчёта и Report ID те же; отчётов столько же, сколько на входе",
         ("test_only_xy_bits_change_and_only_downwards_for_any_report", "test_every_other_interface_is_copied_unchanged", "test_one_report_out_for_every_report_in_through_every_state"),
         "Объединение двух отчётов при переполнении очереди порта (`bridge_merge`) разрешено только когда всё вне X/Y одинаково; в порту это не реализовано"),
    Prop("T6", "Никаких HID feature reports «от помощника», никаких своих usage page наружу, никаких собственных конечных точек", "sim",
         "Ядро не умеет сказать ПК ничего от себя: единственные функции, дающие данные ПК, это `bridge_pc_setup` (из образа мыши) и `bridge_mouse_in` (копия отчёта той же длины). "
         "Тишина мыши даёт тишину моста при любых сообщениях модуля, нажатиях Panic и сбоях. Состояние помощи, телеметрия и статус идут только по SPI к модулю",
         ("test_the_cores_pc_facing_functions_are_exactly_two", "test_a_quiet_mouse_means_a_silent_bridge_through_every_state_and_every_module_message",
          "test_the_mouse_is_asked_nothing_the_pc_did_not_ask"),
         "Мышь видит от моста чтение report-дескрипторов при построении образа (так делает любой хост); других запросов, не заданных ПК, нет"),
    Prop("T7", "При bypass и без питания мышь соединена с ПК электрически напрямую, без перечисления «другого устройства»", "model",
         "Нормально-замкнутые контакты K1/K2 без питания и при любом отказе безопасного домена; в симуляции маршрут bypass даёт ПК перечисление настоящей мыши с тем же транскриптом, что и до адаптера",
         ("test_hard_bypass_is_the_real_mouse_again_with_the_same_transcript", "test_the_pc_only_ever_enumerates_the_identity_of_the_real_mouse"),
         "ПК перенумерует мышь заново (0,2…2 с без ввода, в симуляции 300 мс, допущение); электрика, контакты реле и ток на плате не проверены. «Прямо» значит «тот же кабельный путь через пассивные контакты», а не «как без адаптера» для анализатора сигнала"),
    Prop("T8", "Мягкие переходы (Panic, потеря модуля, TTL, аккорд) не оставляют следа на шине", "sim",
         "`PASSTHRU` не меняет подключения device-порта: нет отключения, нет сброса шины, нет повторного перечисления; отчёт выходит бит в бит",
         ("test_soft_transitions_leave_no_trace_on_the_bus",),
         "Переходы в `ASSIST` и из жёсткого bypass перенумеровывают мышь (это граница T7, раздел 4 документа)"),
    Prop("T9", "Честные границы: что ПК может и не может заметить", "sim",
         "Таблицы «чего ПК не видит» и «что ПК может заметить» создаются из этого модуля; тест проверяет, что документ им соответствует и что границы названы",
         ("test_the_document_is_current_and_names_the_boundaries", "test_every_property_names_tests_that_exist"), ""),
)

# what the PC (and the software on it) can NOT tell apart, with the basis of the claim
INVISIBLE = (
    ("VID/PID, bcdDevice, класс, число и порядок интерфейсов и конечных точек, bMaxPower", "device и configuration дескрипторы отдаются из кэша байт в байт", "sim"),
    ("Строки: производитель, продукт, серийный номер на всех языках, MS OS 0xEE, BOS", "пересылаются мыши вживую", "sim"),
    ("Report-дескрипторы и HID class-дескрипторы", "из кэша байт в байт", "sim"),
    ("Свои USB-интерфейсы, HID-коллекции, usage pages, feature/output-отчёты «помощника»", "их нет: ядро не умеет их создавать", "code"),
    ("Кнопки, колесо, наклон колеса, Report ID, вендорные поля, длина отчёта", "в отчёте движения меняются только биты X/Y", "sim"),
    ("Запросы и ответы SET/GET_REPORT, IDLE, PROTOCOL, вендорные, HID++-подобный канал", "пересылаются мыши вживую, STALL и таймаут зеркалятся", "sim"),
    ("Загрузочный протокол BIOS", "SET_PROTOCOL(0) пересылается, раскладка отчёта переключается", "sim"),
    ("Установленные драйверы, службы, приложения на ПК", "ничего не ставится", "code"),
)

# what the PC or software on it CAN notice, with how big it is and how sure we are
VISIBLE = (
    ("Мышь «отключилась и подключилась» при включении моста и при каждом жёстком bypass", "перенумерация: 0,2…2 с без ввода (в симуляции 300 мс)", "ПК и любое ПО, следящее за hot-plug", "допущение, не измерено"),
    ("Тот же ID, но на шине другое физическое устройство", "анализатор USB, сравнение сигнала, длины кабеля, формы импульсов", "только измерительная техника", "не измерено"),
    ("Скорость соединения", "LS-мышь будет представлена как FS (device-порт не умеет LS), HS-мышь как HS; утилиты показывают скорость", "USBView, lsusb -t, Device Manager", "по устройству порта, на плате не проверено"),
    ("Тайминг отчётов", "+0,5 мс в среднем и до +1 мс на 1 кГц (каскад двух опросов); +62/125 мкс на HS; мышь 125 Гц до +8 мс", "ПО, измеряющее задержку и джиттер опроса", "симуляция каскада сеток"),
    ("Тайминг control-запросов", "пересылка мыши добавляет порядка 1…3 мс на запрос", "ПО, измеряющее перечисление", "оценка"),
    ("Ток с порта ПК", "в режиме моста мышь питается от адаптера, а не от VBUS ПК; ПК объявлено то, что объявила мышь", "хабы с измерением тока по портам, управляемые хабы", "по схеме питания, не измерено"),
    ("Топология и путь питания", "мышь не видна как отдельное устройство на порту адаптера: ПК видит одно устройство на своём порту (хаба между ними нет)", "диспетчеры, показывающие дерево", "по схеме"),
    ("Мышь видит двух хостов подряд", "адаптер перечисляет мышь сам (чтение дескрипторов), затем ПК присылает SET_CONFIGURATION: мышь с состоянием профиля/DPI может сбросить его", "прошивка мыши и её ПО", "не проверено, зависит от мыши"),
    ("Движение указателя", "это и есть помощь: поток X/Y меньше, чем сделала рука", "любое ПО, сравнивающее ввод с движением", "по назначению"),
    ("Дескрипторы, зависящие от состояния", "если у мыши ответ на GET_DESCRIPTOR зависит от режима, ПК получает снятый при построении образа", "редкие мыши с режимами", "не проверено"),
    ("Мыши, которые мост не обслуживает", "изохронные/bulk точки, несколько конфигураций, альтернативные настройки, абсолютный указатель: адаптер не включается, мышь работает напрямую без помощи", "человек: помощи нет", "тесты симулятора"),
)

BY_ID = {p.id: p for p in PROPS}


def _row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def props_table() -> str:
    out = [_row("№", "требование", "статус", "как устроено", "чем доказано", "граница"), _row(*["---"] * 6)]
    for p in PROPS:
        out.append(_row(f"**{p.id}**", p.requirement, STATUS[p.status], p.how, ", ".join(f"`{t}`" for t in p.tests), p.limit or "—"))
    return "\n".join(out)


def invisible_table() -> str:
    out = [_row("что ПК не отличит", "почему", "основание"), _row(*["---"] * 3)]
    out += [_row(a, b, f"`{c}`") for a, b, c in INVISIBLE]
    return "\n".join(out)


def visible_table() -> str:
    out = [_row("что ПК может заметить", "как именно и насколько", "кто заметит", "как известно"), _row(*["---"] * 4)]
    out += [_row(*r) for r in VISIBLE]
    return "\n".join(out)


def status_table() -> str:
    return "\n".join([_row("пометка", "что значит"), _row("---", "---")] + [_row(f"`{k}`", v) for k, v in STATUS.items()])


TABLES = {"props": props_table, "invisible": invisible_table, "visible": visible_table, "status": status_table}
