"""The classes of the assistive detector, and the wall between the two detector worlds.

Only INTERFACE elements are in here. Nothing that stands for a person, a character, an enemy or any target in a game world is, and the
adapter that feeds the ASC refuses any model whose class list is not a subset of this taxonomy (`require_ui_layout`)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UiClass:
    id: int
    name: str
    target: bool  # an element the person may want to reach (False: the pointer, used only to know where the pointer is)
    what: str


UI_CLASSES = (
    UiClass(0, "button", True, "push buttons, toolbar buttons, big clickable tiles"),
    UiClass(1, "icon", True, "icons with or without a caption (desktop icons, tool icons)"),
    UiClass(2, "text_field", True, "input boxes, search boxes, address bars"),
    UiClass(3, "menu_item", True, "menu bar entries, list / dropdown / context-menu rows"),
    UiClass(4, "toggle", True, "checkboxes, radio buttons, switches (with their caption)"),
    UiClass(5, "tab", True, "tab headers"),
    UiClass(6, "window_control", True, "minimize / maximize / close buttons of a window"),
    UiClass(7, "cursor", False, "the mouse pointer (arrow, hand, I-beam)"),
)
NAMES = tuple(c.name for c in UI_CLASSES)
TARGET_NAMES = frozenset(c.name for c in UI_CLASSES if c.target)
ID = {c.name: c.id for c in UI_CLASSES}


class NotAUiModel(ValueError):
    """The model's classes are not the UI taxonomy: it must not feed the assistive chain."""


def require_ui_layout(class_names, n_keypoints: int = 0) -> None:
    """Raise unless `class_names` is a non-empty subset of the UI taxonomy and the model has no keypoints (a pose/people model has both
    keypoints and people classes: it belongs to the research world and is refused here)."""
    names = tuple(class_names)
    if n_keypoints:
        raise NotAUiModel(f"a model with {n_keypoints} keypoints is a pose model, not a UI-element detector")
    bad = [n for n in names if n not in NAMES]
    if not names or bad:
        raise NotAUiModel(f"classes {bad or names} are not interface elements; the assistive chain only accepts {list(NAMES)}")
