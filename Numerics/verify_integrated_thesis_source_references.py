"""Verify notebook PDF locators against one integrated thesis snapshot.

This publication-side gate is intentionally separate from numerical replay.
It requires the thesis root because ``main.tex``, ``main.aux`` and ``main.pdf``
are manuscript artifacts, not numerical inputs shipped in Final Deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tomllib
from typing import Any


HERE = Path(__file__).resolve().parent
REFERENCE_MAP = HERE / "numerical_certification_transfer_markdown.toml"
NAMED_KINDS = frozenset(
    {
        "Chapter",
        "Section",
        "Subsection",
        "Definition",
        "Remark",
        "Lemma",
        "Proposition",
        "Theorem",
        "Corollary",
        "Algorithm",
    }
)
CAPTION_KINDS = frozenset({"Figure", "Table"})


class ReferenceVerificationError(RuntimeError):
    """Raised when a recorded PDF locator drifts from the integrated thesis."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _balanced(text: str, start: int, opening: str, closing: str) -> tuple[str, int]:
    if start >= len(text) or text[start] != opening:
        raise ReferenceVerificationError(
            f"Expected {opening!r} at source offset {start}."
        )
    depth = 0
    for index in range(start, len(text)):
        character = text[index]
        escaped = index > 0 and text[index - 1] == "\\"
        if character == opening and not escaped:
            depth += 1
        elif character == closing and not escaped:
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    raise ReferenceVerificationError(f"Unclosed {opening!r} group at {start}.")


def _parse_aux(aux_path: Path, wanted: set[str]) -> dict[str, list[str]]:
    records: dict[str, list[str]] = {}
    for line in aux_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("\\newlabel{"):
            continue
        label, position = _balanced(line, line.index("{"), "{", "}")
        if label not in wanted:
            continue
        outer, _ = _balanced(line, position, "{", "}")
        fields: list[str] = []
        cursor = 0
        while cursor < len(outer):
            if outer[cursor] == "{":
                field, cursor = _balanced(outer, cursor, "{", "}")
                fields.append(field)
            else:
                cursor += 1
        if len(fields) < 4:
            raise ReferenceVerificationError(f"Malformed AUX record for {label!r}.")
        records[label] = fields
    return records


def _normalise_visible_text(text: str) -> str:
    text = re.sub(
        r"\\texorpdfstring\s*\{\\\((.*?)\\\)\}\{.*?\}",
        lambda match: f"${match.group(1)}$",
        text,
    )
    while re.search(r"\\textcolor\s*\{[^{}]+\}\s*\{([^{}]*)\}", text):
        text = re.sub(
            r"\\textcolor\s*\{[^{}]+\}\s*\{([^{}]*)\}", r"\1", text
        )
    text = re.sub(r"\\nolinkurl\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\color\s*\{[^{}]+\}", "", text)
    text = text.replace(r"\(", "$").replace(r"\)", "$")
    text = text.replace("--", "–").replace("~", " ").replace("`", "")
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def _full_caption(source: str, label: str) -> str:
    marker = f"\\label{{{label}}}"
    label_position = source.find(marker)
    if label_position < 0:
        raise ReferenceVerificationError(f"Caption label {label!r} is absent.")
    caption_position = source.rfind("\\caption", 0, label_position)
    if caption_position < 0:
        raise ReferenceVerificationError(f"Caption for {label!r} is absent.")
    caption_command = (
        "\\captionof"
        if source.startswith("\\captionof", caption_position)
        else "\\caption"
    )
    cursor = caption_position + len(caption_command)
    while cursor < len(source) and source[cursor].isspace():
        cursor += 1
    if caption_command == "\\captionof":
        _, cursor = _balanced(source, cursor, "{", "}")
        while cursor < len(source) and source[cursor].isspace():
            cursor += 1
    if cursor < len(source) and source[cursor] == "[":
        _, cursor = _balanced(source, cursor, "[", "]")
        while cursor < len(source) and source[cursor].isspace():
            cursor += 1
    caption, _ = _balanced(source, cursor, "{", "}")
    return caption


def verify(thesis_root: Path) -> dict[str, Any]:
    """Check hashes, numbers, named titles and visible caption leads."""

    root = thesis_root.resolve()
    payload = tomllib.loads(REFERENCE_MAP.read_text(encoding="utf-8"))
    references = payload["source_references"]
    required_paths = {
        "integrated_thesis_driver_sha256_at_mapping": root
        / payload["integrated_thesis_driver"],
        "integrated_thesis_aux_sha256_at_mapping": root
        / payload["integrated_thesis_aux"],
        "integrated_thesis_pdf_sha256_at_mapping": root
        / payload["integrated_thesis_pdf"],
        "chapter_source_sha256_at_mapping": root / payload["chapter_source"],
        "direct_theory_source_sha256_at_mapping": root
        / payload["direct_theory_source"],
    }
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        raise ReferenceVerificationError(f"Missing thesis artifacts: {missing}.")
    for field, path in required_paths.items():
        actual = _sha256(path)
        if actual != payload[field]:
            raise ReferenceVerificationError(
                f"Integrated-thesis hash drift for {path}: "
                f"recorded={payload[field]}, actual={actual}."
            )

    page_count = payload["integrated_thesis_page_count_at_mapping"]
    if not isinstance(page_count, int) or page_count <= 0:
        raise ReferenceVerificationError("Invalid integrated PDF page-count record.")

    aux_records = _parse_aux(
        root / payload["integrated_thesis_aux"], set(references)
    )
    missing_labels = set(references) - set(aux_records)
    if missing_labels:
        raise ReferenceVerificationError(
            f"Integrated AUX lacks labels {sorted(missing_labels)}."
        )

    numerical_source = (root / payload["chapter_source"]).read_text(
        encoding="utf-8"
    )
    direct_source = (root / payload["direct_theory_source"]).read_text(
        encoding="utf-8"
    )
    number_checks = 0
    named_title_checks = 0
    caption_checks = 0
    for label, reference in references.items():
        aux_number, _, aux_title, _ = aux_records[label][:4]
        if aux_number != reference["number"]:
            raise ReferenceVerificationError(
                f"Number drift for {label!r}: "
                f"recorded={reference['number']!r}, AUX={aux_number!r}."
            )
        number_checks += 1
        kind = reference["kind"]
        if kind in NAMED_KINDS:
            recorded_title = _normalise_visible_text(reference["title"])
            integrated_title = _normalise_visible_text(aux_title)
            if recorded_title != integrated_title:
                raise ReferenceVerificationError(
                    f"Title drift for {label!r}: "
                    f"recorded={recorded_title!r}, AUX={integrated_title!r}."
                )
            named_title_checks += 1
        elif kind in CAPTION_KINDS:
            source = direct_source if label.startswith("legEDMD:") else numerical_source
            visible_caption = _normalise_visible_text(_full_caption(source, label))
            recorded_lead = _normalise_visible_text(reference["title"])
            if not visible_caption.startswith(recorded_lead):
                raise ReferenceVerificationError(
                    f"Visible caption drift for {label!r}: "
                    f"recorded lead={recorded_lead!r}, caption={visible_caption!r}."
                )
            caption_checks += 1

    return {
        "verification_status": "verified",
        "thesis_root": str(root),
        "integrated_pdf_page_count_recorded": page_count,
        "source_reference_count": len(references),
        "number_checks": number_checks,
        "named_title_checks": named_title_checks,
        "caption_lead_checks": caption_checks,
        "equation_title_policy": "number_only",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--thesis-root",
        required=True,
        type=Path,
        help="Root containing main.tex, main.aux, main.pdf and chapters/.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(json.dumps(verify(args.thesis_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
