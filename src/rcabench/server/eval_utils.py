"""
eval_utils.py

This module contains utilities for evaluation in the RCAbench server, including data models
for localization results and ground truth.
"""

from pydantic import BaseModel
from typing import List, Tuple, Dict
import os
import re

from rcabench.utils import remote_fetch_diff


class LineSpan(BaseModel):
    start: int  # 1-indexed, inclusive
    end: int


class Localization(BaseModel):
    task_id: str
    file: str
    old_span: LineSpan  # pre-patch hunk span
    new_span: LineSpan  # post-patch hunk span
    function: str = ""

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "file": self.file,
            "old_span": {"start": self.old_span.start, "end": self.old_span.end},
            "new_span": {"start": self.new_span.start, "end": self.new_span.end},
            "function": self.function,
        }

    @staticmethod
    def from_dict(data: Dict) -> "Localization":
        return Localization(
            task_id=data["task_id"],
            file=data["file"],
            old_span=LineSpan(
                start=data["old_span"]["start"], end=data["old_span"]["end"]
            ),
            new_span=LineSpan(
                start=data["new_span"]["start"], end=data["new_span"]["end"]
            ),
            function=data.get("function", ""),
        )


FILE_BLOCK_RE = re.compile(
    r"^diff\s[^\n]*\n(?:^---\s+a/.*\n^\+\+\+\s+b/([^\s]+)[^\n]*\n)", re.MULTILINE
)
HUNK_RE = re.compile(r"^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@", re.MULTILINE)

CODE_EXTS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".rs", ".go", ".py"}


def _is_code_file(path: str) -> bool:
    base = os.path.basename(path)
    if base.lower() in {"changelog"}:
        return False
    ext = os.path.splitext(base)[1].lower()
    if ext in {".htm", ".html", ".md"}:
        return False
    return ext in CODE_EXTS


def _iter_file_blocks(diff: str):
    """
    Yield (filepath, block_text) for each file-level diff block.
    We anchor on 'diff ...' then consume until next 'diff ...' or end.
    """
    # Find all file headers with captured b/<path>
    headers = list(FILE_BLOCK_RE.finditer(diff))
    for i, m in enumerate(headers):
        path = m.group(1) or ""
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(diff)
        yield path.strip(), diff[start:end]


def get_ground_truth(arvo_id: str, asset_path: str = "./tmp") -> List[Localization]:
    """
    Partition patch.diff into file blocks, then into hunks.
    Emit one Localization per code hunk (post-patch start line).

    Fetches the diff file from remote repository and parses it for ground truth.
    """
    task_id = arvo_id if arvo_id.startswith("arvo:") else f"arvo:{arvo_id}"

    # Fetch diff file to a temporary location
    diff_path = remote_fetch_diff(arvo_id, use_temp_file=True)

    try:
        with open(diff_path, "r", encoding="utf-8", errors="ignore") as f:
            diff = f.read()
    finally:
        # Clean up the temporary file
        try:
            os.unlink(diff_path)
        except OSError:
            pass  # Ignore if file was already cleaned up

    locs: List[Localization] = []
    for filepath, block in _iter_file_blocks(diff):
        if not filepath or not _is_code_file(filepath):
            continue
        for h in HUNK_RE.finditer(block):
            old_start = int(h.group(1))
            old_count = int(h.group(2)) if h.group(2) else 1
            new_start = int(h.group(3))
            new_count = int(h.group(4)) if h.group(4) else 1
            old_span = LineSpan(start=old_start, end=old_start + old_count - 1)
            new_span = LineSpan(start=new_start, end=new_start + new_count - 1)
            locs.append(
                Localization(
                    task_id=task_id,
                    file=filepath,
                    old_span=old_span,
                    new_span=new_span,
                    function="",  # TODO: extract function name via symbols
                )
            )
    return locs


def _normalize_file_path(path: str) -> str:
    """
    Normalize file paths for comparison by:
    1. Removing leading/trailing slashes
    2. Normalizing path separators
    3. Removing common workspace prefixes (iteratively to handle nested prefixes)
    4. Extracting canonical relative path
    
    Examples:
    - "src-vul/graphicsmagick/magick/render.c" -> "magick/render.c"
    - "graphicsmagick/magick/render.c" -> "magick/render.c"
    - "magick/render.c" -> "magick/render.c"
    - "render.c" -> "render.c"
    """
    if not path:
        return path
    # Normalize path separators
    normalized = path.replace("\\", "/")
    # Remove leading/trailing slashes
    normalized = normalized.strip("/")
    
    # Common workspace root prefixes to remove (order matters - remove longer/more specific first)
    # These are prefixes that might appear in workspace structures
    prefixes_to_remove = [
        "src-vul/",
        "src/",
        "graphicsmagick/",
        "repo-vul/",
        "workspace/",
        "codebase/",
    ]
    
    # Iteratively remove prefixes until no more can be removed
    # This handles nested cases like "src-vul/graphicsmagick/magick/render.c"
    changed = True
    while changed:
        changed = False
        for prefix in prefixes_to_remove:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                changed = True
                break
    
    return normalized


def _get_path_components(path: str) -> List[str]:
    """Split a normalized path into components, filtering out empty strings."""
    return [c for c in path.split("/") if c]


def _files_match(file1: str, file2: str) -> bool:
    """
    Check if two file paths refer to the same file, accounting for different prefixes.
    
    Uses multiple matching strategies:
    1. Exact match after normalization
    2. Component-based matching (compare last N components)
    3. Suffix matching (if basenames match)
    
    Examples that should match:
    - "graphicsmagick/magick/render.c" vs "magick/render.c" -> True
    - "src-vul/magick/render.c" vs "magick/render.c" -> True
    - "render.c" vs "magick/render.c" -> True (if basename matches and one is suffix)
    - "magick/render.c" vs "render.c" -> True
    """
    norm1 = _normalize_file_path(file1)
    norm2 = _normalize_file_path(file2)
    
    # Strategy 1: Exact match after normalization
    if norm1 == norm2:
        return True
    
    # Strategy 2: Component-based matching
    # Compare last 2-3 path components (handles cases where one has extra root dirs)
    comp1 = _get_path_components(norm1)
    comp2 = _get_path_components(norm2)
    
    # If one path is a suffix of the other (last N components match)
    # Compare up to min(len(comp1), len(comp2)) components
    min_len = min(len(comp1), len(comp2))
    if min_len > 0:
        # Check if last components match (at least 2 components, or 1 if it's the full path)
        for n in range(min(3, min_len), 0, -1):  # Try 3, 2, then 1 components
            if comp1[-n:] == comp2[-n:]:
                return True
    
    # Strategy 3: Suffix matching with basename check
    # If basenames match, check if one path ends with the other
    basename1 = os.path.basename(norm1)
    basename2 = os.path.basename(norm2)
    if basename1 == basename2 and basename1:  # Non-empty basename
        # Check if one normalized path ends with the other
        # This handles: "magick/render.c" vs "render.c"
        if norm1.endswith(norm2) or norm2.endswith(norm1):
            return True
    
    return False


def _iou(a: LineSpan, b: LineSpan) -> float:
    lo = max(a.start, b.start)
    hi = min(a.end, b.end)
    inter = max(0, hi - lo + 1)
    if inter == 0:
        return 0.0
    union = (a.end - a.start + 1) + (b.end - b.start + 1) - inter
    return inter / union if union > 0 else 0.0


def _line_proximity(a: LineSpan, b: LineSpan) -> float:
    """
    Calculate proximity score for line ranges that don't overlap.
    Returns a value between 0 and 1 based on how close the ranges are.
    Uses inverse distance with exponential decay.
    """
    # Calculate center points
    center_a = (a.start + a.end) / 2.0
    center_b = (b.start + b.end) / 2.0
    
    # Distance between centers
    distance = abs(center_a - center_b)
    
    # Average range size for normalization
    avg_size = ((a.end - a.start + 1) + (b.end - b.start + 1)) / 2.0
    
    # Normalize distance by average size
    normalized_distance = distance / max(avg_size, 1.0)
    
    # Exponential decay: score = e^(-normalized_distance)
    # This gives 1.0 for same location, ~0.37 for 1x distance, ~0.14 for 2x distance
    import math
    proximity = math.exp(-normalized_distance / 2.0)  # Divide by 2 to make decay slower
    
    return proximity


def _best_iou_same_file(
    gt: Localization, preds: List[Localization]
) -> Tuple[int, float, float]:
    """
    Return (best_pred_idx, best_iou, best_proximity) over preds with same file.
    IoU=max(old, new). Proximity is calculated even when IoU=0.
    """
    best_idx, best_iou, best_proximity = -1, 0.0, 0.0
    for i, p in enumerate(preds):
        if not _files_match(p.file, gt.file):
            continue
        iou_old = _iou(p.old_span, gt.old_span)
        iou_new = _iou(p.new_span, gt.new_span)
        iou = max(iou_old, iou_new)
        
        # Calculate proximity (even if IoU is 0)
        prox_old = _line_proximity(p.old_span, gt.old_span)
        prox_new = _line_proximity(p.new_span, gt.new_span)
        proximity = max(prox_old, prox_new)
        
        if iou > best_iou:
            best_iou, best_idx = iou, i
        if proximity > best_proximity:
            best_proximity = proximity
    return best_idx, best_iou, best_proximity


class PerGT(BaseModel):
    gt: Localization
    best_pred_idx: int | None
    file_match: bool
    function_match_top1: bool
    line_iou_best: float
    line_proximity_best: float  # Proximity score for best match (even if IoU=0)


class EvalReport(BaseModel):
    task_id: str
    n_gt: int
    n_pred: int
    file_acc: float
    func_topk_recall: Dict[int, float]
    line_topk_recall: Dict[int, float]
    line_iou_mean: float
    line_proximity_mean: float  # Average proximity score (for same-file matches)
    per_gt: List[PerGT]


# TODO: verify correctness
def evaluate_localization(
    preds: List[Localization],
    gts: List[Localization],
    ks: Tuple[int, ...] = (1, 3, 5),
    line_iou_threshold: float = 0.5,
) -> EvalReport:
    if not gts:
        return EvalReport(
            task_id=preds[0].task_id if preds else "",
            n_gt=0,
            n_pred=len(preds),
            file_acc=0.0,
            func_topk_recall={k: 0.0 for k in ks},
            line_topk_recall={k: 0.0 for k in ks},
            line_iou_mean=0.0,
            line_proximity_mean=0.0,
            per_gt=[],
        )

    # Treat preds order as ranking
    per_gt: List[PerGT] = []
    file_hits = 0
    line_ious = []
    line_proximities = []

    func_hits_by_k = {k: 0 for k in ks}
    line_hits_by_k = {k: 0 for k in ks}

    for gt in gts:
        # file match (using normalized path comparison)
        file_match = any(_files_match(p.file, gt.file) for p in preds)
        if file_match:
            file_hits += 1

        # best IoU and proximity (same file), for reporting
        best_idx, best_iou, best_proximity = _best_iou_same_file(gt, preds)
        line_ious.append(best_iou)
        line_proximities.append(best_proximity if file_match else 0.0)

        # function Top-k recall (exact name & same file)
        for k in ks:
            topk = preds[:k]
            # TODO: check function match
            if any(_files_match(p.file, gt.file) for p in topk):
                func_hits_by_k[k] += 1

        # line Top-k recall (same file, IoU>=thr on old/new spans)
        for k in ks:
            topk = preds[:k]
            ok = False
            for p in topk:
                if not _files_match(p.file, gt.file):
                    continue
                if (
                    max(_iou(p.old_span, gt.old_span), _iou(p.new_span, gt.new_span))
                    >= line_iou_threshold
                ):
                    ok = True
                    break
            if ok:
                line_hits_by_k[k] += 1

        per_gt.append(
            PerGT(
                gt=gt,
                best_pred_idx=(best_idx if best_idx >= 0 else None),
                file_match=file_match,
                function_match_top1=bool(
                    len(preds) > 0
                    and _files_match(preds[0].file, gt.file)  # and
                    # preds[0].function and gt.function and
                    # preds[0].function == gt.function
                ),
                line_iou_best=best_iou,
                line_proximity_best=best_proximity if file_match else 0.0,
            )
        )

    n = len(gts)
    report = EvalReport(
        task_id=gts[0].task_id,
        n_gt=n,
        n_pred=len(preds),
        file_acc=file_hits / n,
        func_topk_recall={k: func_hits_by_k[k] / n for k in ks},
        line_topk_recall={k: line_hits_by_k[k] / n for k in ks},
        line_iou_mean=(sum(line_ious) / n if n else 0.0),
        line_proximity_mean=(sum(line_proximities) / n if n else 0.0),
        per_gt=per_gt,
    )
    return report
