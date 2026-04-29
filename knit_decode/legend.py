from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import cast


RGB = tuple[int, int, int]
_HEX_COLOR_RE = re.compile(r"^#?[0-9A-Fa-f]{6}$")


@dataclass(frozen=True)
class LegendEntry:
    action_id: int
    label: str
    color_hex: str
    rgb: RGB


@dataclass(frozen=True)
class Legend:
    entries: tuple[LegendEntry, ...]
    entries_by_id: dict[int, LegendEntry]
    entries_by_color: dict[RGB, tuple[LegendEntry, ...]]

    @property
    def unique_colors(self) -> tuple[RGB, ...]:
        return tuple(self.entries_by_color.keys())

    @property
    def duplicate_colors(self) -> dict[str, list[int]]:
        duplicates: dict[str, list[int]] = {}
        for rgb, entries in self.entries_by_color.items():
            if len(entries) > 1:
                duplicates[rgb_to_hex(rgb)] = [entry.action_id for entry in entries]
        return duplicates

    def candidate_ids_for_color(self, rgb: RGB) -> list[int]:
        entries = self.entries_by_color.get(rgb, ())
        return [entry.action_id for entry in entries]


def rgb_to_hex(rgb: RGB) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def normalize_hex_color(value: str) -> str:
    if not _HEX_COLOR_RE.match(value):
        raise ValueError(f"Invalid legend color value: {value!r}")
    return value.upper() if value.startswith("#") else f"#{value.upper()}"


def hex_to_rgb(value: str) -> RGB:
    normalized = normalize_hex_color(value)
    return (
        int(normalized[1:3], 16),
        int(normalized[3:5], 16),
        int(normalized[5:7], 16),
    )


def load_legend(path: str | Path) -> Legend:
    legend_path = Path(path)
    payload_raw: object = json.loads(legend_path.read_text(encoding="utf-8"))
    if not isinstance(payload_raw, dict):
        raise ValueError("Legend file must contain a JSON object")
    payload = cast(dict[object, object], payload_raw)

    entries: list[LegendEntry] = []
    entries_by_id: dict[int, LegendEntry] = {}
    entries_by_color: dict[RGB, list[LegendEntry]] = {}

    for label, raw_entries in payload.items():
        if not isinstance(raw_entries, list):
            raise ValueError(f"Legend entries for {label!r} must be a list")
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                raise ValueError(f"Legend entry for {label!r} must be an object")
            raw_entry = cast(dict[object, object], raw_entry)
            action_id = raw_entry.get("id")
            color_hex = raw_entry.get("color")
            if not isinstance(action_id, int):
                raise ValueError(f"Legend entry for {label!r} has invalid id: {action_id!r}")
            if not isinstance(color_hex, str):
                raise ValueError(f"Legend entry for {label!r} has invalid color: {color_hex!r}")
            normalized_hex = normalize_hex_color(color_hex)
            rgb = hex_to_rgb(normalized_hex)
            entry = LegendEntry(
                action_id=action_id,
                label=str(label),
                color_hex=normalized_hex,
                rgb=rgb,
            )
            if action_id in entries_by_id:
                raise ValueError(f"Duplicate legend id detected: {action_id}")
            entries.append(entry)
            entries_by_id[action_id] = entry
            entries_by_color.setdefault(rgb, []).append(entry)

    return Legend(
        entries=tuple(entries),
        entries_by_id=entries_by_id,
        entries_by_color={rgb: tuple(color_entries) for rgb, color_entries in entries_by_color.items()},
    )
