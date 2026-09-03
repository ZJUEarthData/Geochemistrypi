"""Deterministic response views over an immutable CLI artifact index."""

from dataclasses import dataclass
from pathlib import PurePosixPath

from ..api.schemas import ArtifactReference


@dataclass(frozen=True)
class ArtifactViewPartition:
    """Full and canonical views derived without rewriting the artifact index."""

    all_entries: tuple[ArtifactReference, ...]
    canonical_entries: tuple[ArtifactReference, ...]
    summary_mirror_count: int
    summary_mirror_sources: tuple[tuple[str, str], ...]


def _mirror_key(reference: ArtifactReference) -> tuple[tuple[str, ...], str, int, str] | None:
    normalized = reference.relative_path.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    try:
        category_position = parts.index(reference.category)
    except ValueError:
        return None
    if category_position not in {0, 1}:
        return None
    category_tail = parts[category_position + 1 :]
    if not category_tail:
        return None
    # The CLI flattens every summary copy to ``summary/<basename>`` (optionally
    # beneath one aggregate-child scope), even when its source lives below a
    # native directory such as ``artifacts/data`` or
    # ``artifacts/image/model_output``.  A nested *summary* path is therefore a
    # distinct product, while a nested non-summary source remains eligible for
    # the same basename/size/hash uniqueness proof used below.
    if reference.category == "summary" and len(category_tail) != 1:
        return None
    return (
        tuple(parts[:category_position]),
        parts[-1],
        reference.size_bytes,
        reference.sha256,
    )


def partition_artifact_views(entries: tuple[ArtifactReference, ...]) -> ArtifactViewPartition:
    """Suppress only summary copies that have one unambiguous source artifact.

    Equal content alone is insufficient: a summary item is a mirror only when
    one non-summary item in the same aggregate/child scope also has the same
    basename, size, and SHA-256. Requirement-bound summary outputs and ambiguous
    matches remain canonical scientific artifacts.
    """
    candidates: dict[tuple[tuple[str, ...], str, int, str], list[ArtifactReference]] = {}
    for reference in entries:
        if reference.category == "summary":
            continue
        key = _mirror_key(reference)
        if key is not None:
            candidates.setdefault(key, []).append(reference)

    canonical = []
    mirror_sources = []
    for reference in entries:
        if reference.category != "summary" or reference.requirement_id is not None or reference.requirement_ids:
            canonical.append(reference)
            continue
        key = _mirror_key(reference)
        if key is None or len(candidates.get(key, ())) != 1:
            canonical.append(reference)
            continue
        source = candidates[key][0]
        mirror_sources.append((reference.artifact_id, source.artifact_id))

    return ArtifactViewPartition(
        all_entries=entries,
        canonical_entries=tuple(canonical),
        summary_mirror_count=len(mirror_sources),
        summary_mirror_sources=tuple(mirror_sources),
    )
