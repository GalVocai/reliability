"""Build the thesis Results Outline directly from SensorLab plot outputs.

The report builder deliberately consumes generated PNG files rather than a
PowerPoint deck.  A study may span more than one degradation root (the normal
reliability study and the cycles study do in the current data layout), so the
scanner merges *conditions*, not directories.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import pandas as pd

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

from .plotting.overview_plot import (
    _collect_overview_data, generate_overview_plots, summarize_overview_endpoints,
)


_MODE_DIRS = {"JG_n03", "JG_H2", "SE_n04", "STEPS_n07"}
_DEVICE_RE = re.compile(r"^(ESN.+?)(?:_Vbg-?\d+(?:\.\d+)?)?$")


@dataclass(frozen=True)
class ConditionSource:
    """One reliability condition found below a degradation output root."""

    key: str
    label: str
    directory: Path
    degradation_root: Path


@dataclass(frozen=True)
class ReportImage:
    key: str
    label: str
    path: Path

    @property
    def exists(self) -> bool:
        return self.path.is_file()


@dataclass(frozen=True)
class ReportPanel:
    key: str
    title: str
    caption: str
    images: tuple[ReportImage, ...]


@dataclass(frozen=True)
class ReportBuildResult:
    output_path: Path
    manifest_path: Path
    images_added: int
    missing_images: tuple[ReportImage, ...]
    overview_generated: tuple[Path, ...]
    endpoint_summary_path: Path | None = None


_CONDITION_META = {
    "hot_storage": {
        "label": "125 °C Hot Storage",
        "heading": "2.1 High-Temperature Storage",
        "figures_heading": "2.1 Figures",
        "figure_prefix": "2.1",
        "stress_phrase": "storage time",
    },
    "85c_85rh": {
        "label": "85 °C/85% RH",
        "heading": "2.2 85°C/85% RH Test",
        "figures_heading": "2.2 Figures",
        "figure_prefix": "2.2",
        "stress_phrase": "stress time",
    },
    "cycles": {
        "label": "Temperature Cycling",
        "heading": "2.3 Temperature Cycling",
        "figures_heading": "2.3 Figures",
        "figure_prefix": "2.3",
        "stress_phrase": "the number of temperature cycles",
    },
}
_REPORT_ORDER = ("hot_storage", "85c_85rh", "cycles")


def canonical_condition(name: str) -> str:
    """Return the stable report key for a condition directory name."""

    text = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    if "cycle" in text:
        return "cycles"
    if "85" in text and ("rh" in text or "humidity" in text):
        return "85c_85rh"
    if "hot" in text or "125" in text:
        return "hot_storage"
    return text


def _looks_like_condition(path: Path) -> bool:
    if not path.is_dir():
        return False
    children = {p.name for p in path.iterdir() if p.is_dir()}
    return len(children & _MODE_DIRS) >= 2


def _condition_dirs(selected: Path) -> tuple[Path, list[Path]]:
    """Resolve a selected root to (degradation root, condition dirs)."""

    selected = selected.expanduser().resolve()
    if not selected.is_dir():
        raise FileNotFoundError(f"Degradation folder does not exist: {selected}")
    if _looks_like_condition(selected):
        return selected.parent, [selected]
    nested = selected / "degradation"
    if nested.is_dir() and not selected.name.lower() == "degradation":
        selected = nested
    conditions = [p for p in selected.iterdir() if _looks_like_condition(p)]
    if not conditions:
        raise ValueError(
            f"No SensorLab condition folders were found under: {selected}\n"
            "Select a degradation folder whose children contain JG_n03, "
            "JG_H2, SE_n04 and STEPS_n07 outputs."
        )
    return selected, sorted(conditions, key=lambda p: p.name.lower())


def discover_degradation_sources(
    roots: Iterable[str | Path],
) -> dict[str, ConditionSource]:
    """Scan one or more roots and merge their reliability conditions.

    Duplicate canonical conditions are rejected rather than silently choosing
    one study over another (important when both the valid and oven-malfunction
    cycle folders exist).
    """

    found: dict[str, ConditionSource] = {}
    for root_value in roots:
        degradation_root, directories = _condition_dirs(Path(root_value))
        for directory in directories:
            key = canonical_condition(directory.name)
            if key in found and found[key].directory != directory:
                raise ValueError(
                    f"Condition '{key}' was found more than once:\n"
                    f"  {found[key].directory}\n  {directory}\n"
                    "Remove one of the duplicate degradation sources."
                )
            meta = _CONDITION_META.get(key, {})
            found[key] = ConditionSource(
                key=key,
                label=str(meta.get("label", directory.name)),
                directory=directory,
                degradation_root=degradation_root,
            )
    return found


def _devices_in(directory: Path, relative: str, *, strip_vbg: bool = False) -> set[str]:
    folder = directory / relative
    if not folder.is_dir():
        return set()
    devices: set[str] = set()
    for image in folder.glob("*.png"):
        stem = image.stem
        if strip_vbg:
            match = _DEVICE_RE.match(stem)
            if match:
                stem = match.group(1)
        if stem.startswith("ESN"):
            devices.add(stem)
    return devices


def _device_sort_key(device: str) -> tuple[int, int, int, str]:
    match = re.match(r"ESN(\d+)_D(\d+)T(\d+)", device, re.IGNORECASE)
    if not match:
        return (math.inf, math.inf, math.inf, device)
    return (*map(int, match.groups()), device)


def available_report_devices(source: ConditionSource) -> list[str]:
    """Devices having the raw overlays required by the report.

    Prefer the intersection of JG, SE and STEPS.  If an older degradation run
    omitted one family, return the union so the user can still build a report
    with explicit missing placeholders.
    """

    groups = [
        _devices_in(source.directory, "JG_n03/sweeps", strip_vbg=True),
        _devices_in(source.directory, "SE_n04/sensing"),
        _devices_in(source.directory, "STEPS_n07/sensing"),
    ]
    nonempty = [group for group in groups if group]
    if not nonempty:
        return []
    common = set.intersection(*nonempty)
    devices = common if common else set.union(*nonempty)
    return sorted(devices, key=_device_sort_key)


def _first_candidate(base: Path, relatives: Sequence[str]) -> Path:
    for relative in relatives:
        candidate = base / relative
        if candidate.is_file():
            return candidate
    return base / relatives[0]


def _slot(base: Path, key: str, label: str, *relatives: str) -> ReportImage:
    return ReportImage(key, label, _first_candidate(base, relatives))


def condition_panels(source: ConditionSource, device: str) -> tuple[ReportPanel, ...]:
    """Resolve the four standard panel groups for one condition."""

    meta = _CONDITION_META[source.key]
    prefix = str(meta["figure_prefix"])
    stress = str(meta["stress_phrase"])
    base = source.directory
    sensitivity = (
        "STEPS_n07/sensitivity_vs_cycle.png"
        if source.key == "cycles"
        else "STEPS_n07/sensitivity_vs_day.png"
    )

    electrical = ReportPanel(
        "electrical",
        "Electrical characteristics evolution (JG sweeps) & Vth drift",
        f"Figure {prefix}.1 – Electrical characteristics evolution (JG sweeps) "
        f"and Vth drift: (a) correlated to back-gate 0 V; (b) correlated to "
        f"back-gate 4 V; (c) JG curves collapsed to baseline, back-gate 0 V; "
        f"(d) JG curves collapsed to baseline, back-gate 4 V; (e) Vth drift "
        f"as a function of {stress}.",
        (
            _slot(base, "jg_vbg0", "JG evolution, Vbg=0 V", f"JG_n03/sweeps/{device}_Vbg0.png"),
            _slot(base, "jg_vbg4", "JG evolution, Vbg=4 V", f"JG_n03/sweeps/{device}_Vbg4.png"),
            _slot(base, "jg_collapse_vbg0", "JG collapse, Vbg=0 V",
                  f"JG_n03/collapsed/{device}_Vbg0.png",
                  f"JG_n03/collapse/{device}_Vbg0.png"),
            _slot(base, "jg_collapse_vbg4", "JG collapse, Vbg=4 V",
                  f"JG_n03/collapsed/{device}_Vbg4.png",
                  f"JG_n03/collapse/{device}_Vbg4.png"),
            _slot(base, "delta_vth", "Vth drift", "JG_n03/trends/average/delta_Vth.png"),
        ),
    )
    jg_h2 = ReportPanel(
        "jg_h2",
        "Hydrogen sensing — JG sweep",
        f"Figure {prefix}.2 – Hydrogen sensing, extracted from JG sweeps before "
        f"and during/after 100 ppm H2 exposure: (a) JG sweep overlay, baseline; "
        f"(b) JG sweep overlay, end of test; (c) first ΔVth as a function of "
        f"{stress}; (d) response at Vth (maximum response) as a function of "
        f"{stress}; (e) second ΔVth after H2 recovery as a function of {stress}.",
        (
            _slot(base, "jg_h2_baseline", "JG H2 overlay, baseline",
                  f"JG_H2/overlays/{device}_baseline.png",
                  f"JG_H2/overlays/baseline/{device}.png"),
            _slot(base, "jg_h2_end", "JG H2 overlay, end of test",
                  f"JG_H2/overlays/{device}_end.png",
                  f"JG_H2/overlays/end/{device}.png"),
            _slot(base, "first_delta_vth", "First ΔVth", "JG_H2/trends/average/delta_Vth_H2.png"),
            _slot(base, "response_at_vth", "Response at Vth",
                  "JG_H2/trends/average/resp_at_Vth_H2_pct.png"),
            _slot(base, "second_delta_vth", "Second ΔVth",
                  "JG_H2/trends/average/delta_Vth_recovery.png"),
        ),
    )
    id_t = ReportPanel(
        "id_t",
        "Hydrogen sensing — Id–t (time-dependent)",
        f"Figure {prefix}.3 – Hydrogen sensing, extracted from the Id–t "
        f"measurement under 100/1000 ppm H2: (a) transient sensing overlay; "
        f"(b) absolute current increase (ΔId) as a function of {stress}; "
        f"(c) response time (t90) as a function of {stress}.",
        (
            _slot(base, "se_overlay", "Transient sensing overlay", f"SE_n04/sensing/{device}.png"),
            _slot(base, "absolute_current", "Absolute current increase",
                  "SE_n04/trends/average/response_delta.png"),
            _slot(base, "response_time", "Response time",
                  "SE_n04/trends/average/t90_response_s.png"),
        ),
    )
    steps = ReportPanel(
        "steps",
        "Hydrogen concentration steps",
        f"Figure {prefix}.4 – Hydrogen concentration steps: (a) steps overlay; "
        f"(b) ΔResponse as a function of H2 concentration; "
        f"(c) sensitivity as a function of {stress}.",
        (
            _slot(base, "steps_overlay", "Hydrogen concentration steps", f"STEPS_n07/sensing/{device}.png"),
            _slot(base, "response_vs_ppm", "ΔResponse vs H2 concentration", "STEPS_n07/response_vs_ppm.png"),
            _slot(base, "sensitivity", "Sensitivity", sensitivity,
                  "STEPS_n07/trends/average/sensitivity_pct_per_dec.png"),
        ),
    )
    return electrical, jg_h2, id_t, steps


def _find_supplemental(root: Path, relatives: Sequence[str]) -> Path:
    for relative in relatives:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    # Accept either the project output root or publication_plots itself.
    for relative in relatives:
        candidate = root / Path(relative).name
        if candidate.is_file():
            return candidate
    return root / relatives[0]


def overview_panel(supplemental_root: str | Path) -> ReportPanel:
    root = Path(supplemental_root).expanduser().resolve()
    specs = (
        ("overview_delta_vth", "Stress-induced ΔVth",
         ("publication_plots/regular/stress_induced_deltaVTh.png", "regular/stress_induced_deltaVTh.png")),
        ("overview_h2_delta_vth", "Normalized H2-induced ΔVth",
         ("publication_plots/normalized/normalized_H2_induced_deltaVTh.png", "normalized/normalized_H2_induced_deltaVTh.png")),
        ("overview_max_response", "Normalized maximum H2 response",
         ("publication_plots/normalized/normalized_maximum_H2_response_vs_stress_time.png",
          "normalized/normalized_maximum_H2_response_vs_stress_time.png")),
        ("overview_recovery_delta_vth", "Recovery ΔVth",
         ("publication_plots/regular/recovery_deltaVTh_vs_stress_time.png",
          "regular/recovery_deltaVTh_vs_stress_time.png")),
        ("overview_sensitivity", "Normalized sensitivity",
         ("publication_plots/normalized/normalized_sensitivity_vs_stress_time.png",
          "normalized/normalized_sensitivity_vs_stress_time.png")),
        ("overview_response_time", "Normalized H2 response time",
         ("publication_plots/normalized/normalized_H2_response_time_vs_stress_time.png",
          "normalized/normalized_H2_response_time_vs_stress_time.png")),
    )
    images = tuple(
        ReportImage(key, label, _find_supplemental(root, relatives))
        for key, label, relatives in specs
    )
    return ReportPanel(
        "overview",
        "Overview – Comparison Across Reliability Tests",
        "Figure 2.0 – Overview comparing the three reliability tests "
        "(125 °C hot storage, 85 °C/85 %RH, and temperature cycling) as a "
        "function of stress time: (a) ΔVth; (b) normalized hydrogen-induced "
        "shift (first ΔVth, 100 ppm H2); (c) normalized maximum response "
        "(response at Vth, 100 ppm H2); (d) ΔVth after recovery (second ΔVth); "
        "(e) normalized sensitivity; (f) normalized response time.",
        images,
    )


def _paragraph_by_text(document: Document, text: str):
    for paragraph in document.paragraphs:
        if paragraph.text.strip() == text:
            return paragraph
    raise ValueError(f"Required outline heading was not found: {text!r}")


def _remove_blank_paragraphs_before(anchor) -> None:
    previous = anchor._p.getprevious()
    while previous is not None:
        text = "".join(previous.itertext()).strip()
        if text:
            break
        before = previous.getprevious()
        previous.getparent().remove(previous)
        previous = before


def _insert_text(anchor, text: str, *, bold: bool = False,
                 italic: bool = False, size: float = 10.0):
    paragraph = anchor.insert_paragraph_before()
    run = paragraph.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    return paragraph


def _remove_table_borders(table) -> None:
    props = table._tbl.tblPr
    borders = props.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        props.append(borders)
    for edge_name in ("top", "left", "bottom", "right", "insideH", "insideV"):
        edge = borders.find(qn(f"w:{edge_name}"))
        if edge is None:
            edge = OxmlElement(f"w:{edge_name}")
            borders.append(edge)
        edge.set(qn("w:val"), "nil")


def _prevent_row_split(row) -> None:
    props = row._tr.get_or_add_trPr()
    if props.find(qn("w:cantSplit")) is None:
        props.append(OxmlElement("w:cantSplit"))


def _insert_panel_table(document: Document, anchor, panel: ReportPanel,
                        missing: list[ReportImage]) -> int:
    columns = 3
    rows = math.ceil(len(panel.images) / columns)
    table = document.add_table(rows=rows, cols=columns)
    table.autofit = False
    _remove_table_borders(table)
    added = 0
    for row in table.rows:
        _prevent_row_split(row)
    for index, image in enumerate(panel.images):
        cell = table.cell(index // columns, index % columns)
        picture_paragraph = cell.paragraphs[0]
        picture_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if image.exists:
            picture_paragraph.add_run().add_picture(str(image.path), width=Inches(1.94))
            added += 1
        else:
            missing.append(image)
            run = picture_paragraph.add_run(f"[Missing plot]\n{image.label}")
            run.bold = True
            run.font.size = Pt(9)
        label_paragraph = cell.add_paragraph(f"({chr(ord('a') + index)})")
        label_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        label_paragraph.runs[0].font.size = Pt(8)
    # Move the table from the document end to immediately before the anchor.
    anchor._p.addprevious(table._tbl)
    return added


def _insert_panel(document: Document, anchor, panel: ReportPanel,
                  missing: list[ReportImage]) -> int:
    title = _insert_text(anchor, panel.title, bold=True, size=10)
    title.paragraph_format.keep_with_next = True
    added = _insert_panel_table(document, anchor, panel, missing)
    caption = _insert_text(anchor, panel.caption, italic=True, size=9)
    caption.paragraph_format.space_after = Pt(6)
    return added


def _manifest_panel(panel: ReportPanel) -> dict:
    return {
        "key": panel.key,
        "title": panel.title,
        "caption": panel.caption,
        "images": [
            {
                "key": image.key,
                "label": image.label,
                "path": str(image.path),
                "exists": image.exists,
            }
            for image in panel.images
        ],
    }


def _insert_endpoint_summary(document: Document, anchor, summary: pd.DataFrame) -> None:
    heading = _insert_text(anchor, "Final-checkpoint numerical summary", bold=True, size=11)
    heading.paragraph_format.keep_with_next = True
    _insert_text(anchor,
                 "Values use exactly the population means plotted in the overview. "
                 "Each row uses its own last available day and the earliest plotted day as reference. "
                 "Change (%) = 100 x (last/reference - 1). A 0.60 normalized final value means a "
                 "40% decrease. Response-time increases mean slower response. Signed voltage shifts "
                 "are not assigned a performance-loss score; percentage change is N/A at zero reference. "
                 "There is no combined performance score across metrics.", size=9)
    for key in _REPORT_ORDER:
        rows = summary[summary.condition == key]
        title = _insert_text(anchor, str(_CONDITION_META[key]["label"]), bold=True, size=10)
        title.paragraph_format.keep_with_next = True
        table = document.add_table(rows=1, cols=6)
        headers = ("Metric (unit)", "Reference value (day)", "Last day", "Last value", "Change (%)", "Interpretation")
        for cell, label in zip(table.rows[0].cells, headers):
            cell.text = label
        repeat_header = OxmlElement("w:tblHeader")
        table.rows[0]._tr.get_or_add_trPr().append(repeat_header)
        for row in rows.itertuples():
            change = "N/A" if pd.isna(row.relative_change_pct) else f"{row.relative_change_pct:+.1f}%"
            values = (f"{row.metric_label} ({row.unit})",
                      f"{row.baseline_value:.4g} (day {row.baseline_day:g})",
                      f"{row.last_day:g}", f"{row.last_value:.4g}", change, row.interpretation)
            for cell, value in zip(table.add_row().cells, values):
                cell.text = value
        for row in table.rows:
            _prevent_row_split(row)
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(8)
        anchor._p.addprevious(table._tbl)


def build_results_report(
    outline_path: str | Path,
    degradation_roots: Iterable[str | Path],
    supplemental_root: str | Path,
    output_path: str | Path,
    representative_devices: Mapping[str, str] | None = None,
    *,
    overwrite: bool = False,
) -> ReportBuildResult:
    """Create a graph-populated Results Outline and a JSON manifest."""

    outline = Path(outline_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not outline.is_file():
        raise FileNotFoundError(f"Outline document does not exist: {outline}")
    if outline == output:
        raise ValueError("Output must be a new document; the clean outline is never overwritten.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output document already exists: {output}")

    roots = [Path(root).expanduser().resolve() for root in degradation_roots]
    sources = discover_degradation_sources(roots)
    missing_conditions = [key for key in _REPORT_ORDER if key not in sources]
    if missing_conditions:
        raise ValueError("Missing required report conditions: " + ", ".join(missing_conditions))

    devices: dict[str, str] = {}
    requested = dict(representative_devices or {})
    for key in _REPORT_ORDER:
        choices = available_report_devices(sources[key])
        selected = requested.get(key, "")
        if selected and selected not in choices:
            raise ValueError(f"Representative device {selected!r} is not available for {sources[key].label}.")
        if not selected:
            if not choices:
                raise ValueError(f"No report devices were found for {sources[key].label}.")
            selected = choices[0]
        devices[key] = selected

    document = Document(str(outline))
    if document.tables:
        raise ValueError("The selected outline already contains tables. Select the clean Results outline.docx.")
    section_anchors = {
        key: _paragraph_by_text(document, str(_CONDITION_META[key]["heading"]))
        for key in _REPORT_ORDER
    }
    end_anchor = document.paragraphs[-1]
    if end_anchor.text.strip():
        end_anchor = document.add_paragraph()

    directories = {key: sources[key].directory for key in _REPORT_ORDER}
    # Collect once: plots and numerical summaries must use the exact same data.
    overview_data = _collect_overview_data(directories)
    endpoint_summary = summarize_overview_endpoints(overview_data)
    overview_generated = generate_overview_plots(directories, supplemental_root, data=overview_data)
    overview = overview_panel(supplemental_root)
    panels_by_condition = {
        key: condition_panels(sources[key], devices[key]) for key in _REPORT_ORDER
    }
    missing_images: list[ReportImage] = []
    images_added = 0

    hot_anchor = section_anchors["hot_storage"]
    _remove_blank_paragraphs_before(hot_anchor)
    images_added += _insert_panel(document, hot_anchor, overview, missing_images)
    _insert_endpoint_summary(document, hot_anchor, endpoint_summary)

    next_anchor = {
        "hot_storage": section_anchors["85c_85rh"],
        "85c_85rh": section_anchors["cycles"],
        "cycles": end_anchor,
    }
    for key in _REPORT_ORDER:
        anchor = next_anchor[key]
        _remove_blank_paragraphs_before(anchor)
        heading = _insert_text(anchor, str(_CONDITION_META[key]["figures_heading"]), bold=True, size=11)
        heading.paragraph_format.keep_with_next = True
        for panel in panels_by_condition[key]:
            images_added += _insert_panel(document, anchor, panel, missing_images)

    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(output))
    endpoint_summary_path = output.with_name(output.stem + ".endpoints.xlsx")
    with pd.ExcelWriter(endpoint_summary_path, engine="openpyxl") as writer:
        endpoint_summary.to_excel(writer, sheet_name="Final checkpoint", index=False)
        plotted_rows = []
        for metric, series in overview_data.items():
            for condition, frame in series.items():
                plotted_rows.append(frame.assign(metric=metric, condition=condition))
        pd.concat(plotted_rows, ignore_index=True).to_excel(writer, sheet_name="Plot source values", index=False)
        for sheet in writer.sheets.values():
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                sheet.column_dimensions[column[0].column_letter].width = min(
                    55, max(12, max(len(str(cell.value or "")) for cell in column) + 2))
    manifest_path = output.with_name(output.stem + ".manifest.json")
    manifest = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "outline": str(outline),
        "output": str(output),
        "degradation_roots": [str(root) for root in roots],
        "supplemental_root": str(Path(supplemental_root).expanduser().resolve()),
        "overview_generated": [str(path) for path in overview_generated],
        "overview": _manifest_panel(overview),
        "endpoint_summary_path": str(endpoint_summary_path),
        "endpoint_summary": json.loads(endpoint_summary.to_json(orient="records")),
        "conditions": {
            key: {
                "label": sources[key].label,
                "source": str(sources[key].directory),
                "representative_device": devices[key],
                "panels": [_manifest_panel(panel) for panel in panels_by_condition[key]],
            }
            for key in _REPORT_ORDER
        },
        "images_added": images_added,
        "missing_images": [
            {"key": image.key, "label": image.label, "path": str(image.path)}
            for image in missing_images
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return ReportBuildResult(
        output, manifest_path, images_added, tuple(missing_images), overview_generated, endpoint_summary_path
    )
