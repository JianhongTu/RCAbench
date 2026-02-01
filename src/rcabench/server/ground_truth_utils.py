"""
ground_truth_utils.py

This module contains all utilities for extracting and processing ground truth data
from patch files and source code for RCA evaluation.

Functions:
- get_ground_truth: Main function to extract ground truth from patch.diff
- derive_function_name: Use tree-sitter to extract function names from source files
- augment_ground_truth_with_functions: Add function names to ground truth localizations
"""

import os
import re
import glob
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

# Optional tree-sitter imports - fallback to regex if not available
try:
    from tree_sitter import Language, Parser
    import tree_sitter_c
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    Language = None
    Parser = None
    tree_sitter_c = None

from rcabench.utils import remote_fetch_diff
from rcabench.server.eval_utils import Localization, LineSpan

# Regex patterns for parsing patch files
# Match: diff line, optional index line, then --- a/... and +++ b/... lines
# The filepath is captured from the +++ b/ line
FILE_BLOCK_RE = re.compile(
    r"^diff\s[^\n]*\n(?:^index\s[^\n]*\n)?^---\s+a/.*\n^\+\+\+\s+b/([^\s]+)[^\n]*\n", re.MULTILINE
)
# Hunk regex: captures line numbers and function name from hunk header
# Format: @@ -old_start,old_count +new_start,new_count @@ function_name
HUNK_RE = re.compile(r"^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@\s*(.*)$", re.MULTILINE)

CODE_EXTS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".rs", ".go", ".py"}


def _is_code_file(path: str) -> bool:
    """Check if a file is a code file based on extension."""
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


def _extract_hunk_changes(block: str, hunk_match: re.Match) -> Tuple[List[str], List[str], List[str], List[int]]:
    """
    Extract the removed lines, context, and line numbers from a hunk.
    
    Returns:
        (context_before, removed_lines, context_after, removed_line_numbers)
        removed_line_numbers: List of line numbers (1-indexed) where removed lines are in the source
    """
    hunk_start = hunk_match.end()  # Start after the @@ line
    hunk_text = block[hunk_start:]
    
    # Find the next @@ or end of block
    next_hunk = re.search(r'^@@', hunk_text, re.MULTILINE)
    if next_hunk:
        hunk_text = hunk_text[:next_hunk.start()]
    
    lines = hunk_text.split('\n')
    
    context_before = []
    removed_lines = []
    context_after = []
    removed_line_numbers = []
    in_removed = False
    after_removed = False
    
    # Get the old_start line number from the hunk header
    old_start = int(hunk_match.group(1))
    old_count = int(hunk_match.group(2)) if hunk_match.group(2) else 1
    current_line = old_start
    
    for line in lines:
        if not line:
            continue
        
        marker = line[0] if line else ''
        content = line[1:] if len(line) > 1 else ''
        
        if marker == '-':
            # This is a removed line (vulnerable code)
            removed_lines.append(content)
            removed_line_numbers.append(current_line)
            current_line += 1
            in_removed = True
        elif marker == '+':
            # Skip added lines (they don't exist in the old file)
            if in_removed and not after_removed:
                after_removed = True
            # Don't increment current_line for added lines
            continue
        elif marker == ' ':
            # Context line (exists in both old and new)
            if not in_removed:
                # Before the removed lines
                context_before.append(content)
            elif after_removed:
                # After the removed lines
                context_after.append(content)
            current_line += 1
        else:
            # Unknown marker, treat as context
            if not in_removed:
                context_before.append(content)
            elif after_removed:
                context_after.append(content)
            current_line += 1
    
    # Limit context to last 3 lines before and first 3 lines after
    context_before = context_before[-3:] if len(context_before) > 3 else context_before
    context_after = context_after[:3] if len(context_after) > 3 else context_after
    
    return context_before, removed_lines, context_after, removed_line_numbers


def _find_vulnerable_lines_in_file(
    filepath: str, 
    removed_lines: List[str], 
    context_before: List[str], 
    context_after: List[str], 
    asset_path: str = "./tmp"
) -> List[int]:
    """
    Search for the vulnerable lines in the actual source file.
    Uses context lines for fuzzy matching since line numbers in patches may be stale.
    
    Args:
        filepath: Relative path to the source file (e.g., "magick/utility.c")
        removed_lines: Lines that were removed in the patch (the vulnerable code)
        context_before: Context lines before the change
        context_after: Context lines after the change
        asset_path: Base path for task workspace
    
    Returns:
        List of actual line numbers (1-indexed) where the vulnerable code was found
    """
    # Try to locate the actual source file in the workspace
    # Common patterns: asset_path/workspace/src-vul/<filepath>
    possible_paths = [
        os.path.join(asset_path, "workspace", "src-vul", filepath),
        os.path.join(asset_path, "src-vul", filepath),
        os.path.join(asset_path, filepath),
    ]
    
    source_file = None
    for path in possible_paths:
        if os.path.isfile(path):
            source_file = path
            break
    
    if not source_file or not os.path.isfile(source_file):
        # Can't find the file, return empty list
        return []
    
    try:
        with open(source_file, "r", encoding="utf-8", errors="ignore") as f:
            source_lines = f.readlines()
    except Exception:
        return []
    
    # Strip trailing newlines for comparison
    source_lines = [line.rstrip("\n\r") for line in source_lines]
    removed_lines = [line.rstrip("\n\r") for line in removed_lines]
    context_before = [line.rstrip("\n\r") for line in context_before]
    context_after = [line.rstrip("\n\r") for line in context_after]
    
    found_lines = []
    
    # Search for the pattern in the source file
    # We look for: context_before + removed_lines + context_after
    pattern_len = len(context_before) + len(removed_lines) + len(context_after)
    
    for i in range(len(source_lines) - pattern_len + 1):
        # Check if context_before matches
        if context_before:
            before_match = all(
                source_lines[i + j].strip() == context_before[j].strip()
                for j in range(len(context_before))
            )
            if not before_match:
                continue
        
        # Check if removed_lines match
        removed_start = i + len(context_before)
        removed_match = all(
            source_lines[removed_start + j].strip() == removed_lines[j].strip()
            for j in range(len(removed_lines))
        )
        if not removed_match:
            continue
        
        # Check if context_after matches
        if context_after:
            after_start = removed_start + len(removed_lines)
            after_match = all(
                source_lines[after_start + j].strip() == context_after[j].strip()
                for j in range(len(context_after))
            )
            if not after_match:
                continue
        
        # Found a match! Record the line numbers of the removed lines (1-indexed)
        for j in range(len(removed_lines)):
            found_lines.append(removed_start + j + 1)  # +1 for 1-indexed
    
    return found_lines


def _merge_overlapping_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merge overlapping intervals.
    
    Args:
        intervals: List of (start, end) tuples representing intervals
    
    Returns:
        List of merged (start, end) tuples with no overlaps
    """
    if not intervals:
        return []
    
    # Sort intervals by start position
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    
    for current in sorted_intervals[1:]:
        last = merged[-1]
        # If current interval overlaps with or is adjacent to the last merged interval
        if current[0] <= last[1] + 1:  # +1 for adjacent intervals (e.g., [1,5] and [6,10] -> [1,10])
            # Merge them by extending the end
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            # No overlap, add as new interval
            merged.append(current)
    
    return merged


def get_ground_truth(arvo_id: str, asset_path: str = "./tmp") -> List[Localization]:
    """
    Partition patch.diff into file blocks, then into hunks.
    For each hunk, search for the actual vulnerable lines in the source file.
    Emit one Localization per code hunk with the actual line numbers found.

    Fetches the diff file from remote repository and parses it for ground truth.
    
    Args:
        arvo_id: ARVO task ID
        asset_path: Base path for task workspace (e.g., "tmp/arvo_15707-xxx/")
    
    Returns:
        List of Localization objects with ground truth data
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
    
    # asset_path should be the agent directory (e.g., tmp/arvo_15707-xxx/)
    # We'll use it directly as the base for finding source files
    actual_asset_path = asset_path
    
    for filepath, block in _iter_file_blocks(diff):
        if not filepath or not _is_code_file(filepath):
            continue
        
        for h in HUNK_RE.finditer(block):
            old_start = int(h.group(1))
            old_count = int(h.group(2)) if h.group(2) else 1
            new_start = int(h.group(3))
            new_count = int(h.group(4)) if h.group(4) else 1
            
            # Extract function name from hunk header (group 5)
            function_name_raw = h.group(5).strip() if h.group(5) else ""
            # Extract just the function name (remove return type, parameters, etc.)
            # Pattern: "static void FunctionName(...)" -> "FunctionName"
            function_name = ""
            if function_name_raw:
                # Try to extract function name from the header
                # Common patterns: "static void FunctionName(...)", "FunctionName(...)", etc.
                func_match = re.search(r'(\w+)\s*\(', function_name_raw)
                if func_match:
                    function_name = func_match.group(1)
            
            # Extract the removed lines, context, and line numbers from this hunk
            context_before, removed_lines, context_after, removed_line_numbers = _extract_hunk_changes(block, h)
            
            # Search for the actual line numbers in the source file
            actual_lines = _find_vulnerable_lines_in_file(
                filepath, removed_lines, context_before, context_after, actual_asset_path
            )
            
            # Use actual lines if found, otherwise fall back to diff line numbers
            if actual_lines:
                # Use the actual found lines from the source file
                line_numbers_to_use = actual_lines
            elif removed_line_numbers:
                # Fallback to diff line numbers if source file search failed
                line_numbers_to_use = removed_line_numbers
            else:
                # Final fallback: create a range from patch line numbers
                # But we still want individual lines, so create one per removed line
                line_numbers_to_use = list(range(old_start, old_start + len(removed_lines)))
            
            # Create intervals with ±5 buffer for each removed line
            buffer = 5
            for line_num in line_numbers_to_use:
                # Each line gets a range with ±5 buffer
                line_start_buffered = max(1, line_num - buffer)  # Don't go below line 1
                line_end_buffered = line_num + buffer
                
                locs.append(
                    Localization(
                        task_id=task_id,
                        file=filepath,
                        old_span=LineSpan(start=line_start_buffered, end=line_end_buffered),
                        new_span=LineSpan(start=line_start_buffered, end=line_end_buffered),
                        function=function_name,
                    )
                )
    
    # Now merge overlapping intervals for the same file and function
    # Group by (file, function)
    grouped = defaultdict(list)
    for loc in locs:
        key = (loc.file, loc.function)
        grouped[key].append(loc)
    
    # Merge overlapping intervals within each group
    merged_locs = []
    for (file, function), group_locs in grouped.items():
        # Extract intervals from this group
        intervals = [(loc.old_span.start, loc.old_span.end) for loc in group_locs]
        
        # Merge overlapping intervals
        merged_intervals = _merge_overlapping_intervals(intervals)
        
        # Create one Localization per merged interval
        for start, end in merged_intervals:
            merged_locs.append(
                Localization(
                    task_id=task_id,
                    file=file,
                    old_span=LineSpan(start=start, end=end),
                    new_span=LineSpan(start=start, end=end),
                    function=function,
                )
            )
    
    return merged_locs


def derive_function_name(file_path: Path, line_number: int, trace_only: bool = False) -> str:
    """
    Use tree-sitter to derive the function name containing the given line number.
    Falls back to regex-based search if tree-sitter fails.
    
    Args:
        file_path: Path to the source file
        line_number: Line number (1-indexed) to find function for
        trace_only: Whether to output trace logs
    
    Returns:
        Function name as string, or empty string if not found
    """
    try:
        # Read the source file
        with open(file_path, 'rb') as f:
            source_code = f.read()
        
        lines = source_code.split(b'\n')
        if line_number > len(lines):
            return ""
        
        # Try tree-sitter first if available
        if TREE_SITTER_AVAILABLE:
            try:
                # Set up tree-sitter for C
                C_LANGUAGE = Language(tree_sitter_c.language())
                parser = Parser(C_LANGUAGE)
                
                # Parse the file
                tree = parser.parse(source_code)
                
                # Calculate byte offset for the start of the target line
                byte_offset = sum(len(line) + 1 for line in lines[:line_number - 1])  # +1 for newline
                target_line = line_number - 1  # 0-indexed for array access
                
                # Recursively search for function_definition node containing this position
                def find_function_at_position(node, position, depth=0):
                    # Check if this position is within this node
                    if not (node.start_byte <= position <= node.end_byte):
                        return None
                    
                    # If this is a function_definition, we found it
                    if node.type == 'function_definition':
                        # Extract the function name from the declarator
                        declarator = None
                        for child in node.children:
                            if child.type == 'function_declarator':
                                declarator = child
                                break
                        
                        if declarator:
                            # Find the identifier (function name) in the declarator
                            for child in declarator.children:
                                if child.type == 'identifier':
                                    func_name = source_code[child.start_byte:child.end_byte].decode('utf-8')
                                    return func_name
                                # Handle pointer declarators: type *function_name(...)
                                elif child.type == 'pointer_declarator':
                                    for subchild in child.children:
                                        if subchild.type == 'identifier':
                                            func_name = source_code[subchild.start_byte:subchild.end_byte].decode('utf-8')
                                            return func_name
                        
                        # If we found the function but couldn't extract name, return empty
                        return ""
                    
                    # Recursively search children
                    for child in node.children:
                        result = find_function_at_position(child, position, depth + 1)
                        if result is not None:
                            return result
                    
                    return None
                
                result = find_function_at_position(tree.root_node, byte_offset)
                
                # If tree-sitter found a function, return it
                if result:
                    return result
            except Exception:
                # Tree-sitter failed, fall through to regex fallback
                pass
        
        # FALLBACK: Tree-sitter not available or failed (likely due to parsing errors in complex C code)
        # Use regex to search backwards for function definition
        import re
        
        # C function definition pattern: return_type function_name(...) {
        # Must start at or near column 0 (max 2 spaces indent) to be a real function
        func_pattern = re.compile(
            r'^[ ]{0,2}'  # Max 2 spaces of indentation
            r'(?:static\s+|inline\s+|extern\s+|const\s+)*'  # Optional qualifiers
            r'(?:[\w]+\s+)+'  # Return type words (e.g., "unsigned int"), at least one
            r'[\*\s]*'  # Optional pointers and spaces
            r'(\w+)\s*\('  # Capture: function_name followed by (
            r'[^{;]*?'  # Parameters (non-greedy, no { or ; before closing)
            r'\)\s*(?:__attribute__[^{]*)?'  # Closing paren, optional attributes
            r'\s*\{',  # Opening brace
            re.MULTILINE
        )
        
        # C keywords to filter out (not function names)
        c_keywords = {'if', 'while', 'for', 'switch', 'return', 'sizeof', 'typeof', 'else'}
        
        # Search backwards from target line
        text_lines = [line.decode('utf-8', errors='ignore') for line in lines]
        
        # Look backwards up to 1000 lines or start of file
        search_start = max(0, line_number - 1000)
        search_text = '\n'.join(text_lines[search_start:line_number])
        
        # Find all function definitions before the target line
        matches = [m for m in func_pattern.finditer(search_text) 
                   if m.group(1).lower() not in c_keywords]
        
        if matches:
            # Return the last (closest) match
            return matches[-1].group(1)
        
        return ""
        
    except Exception as e:
        # Silently return empty string on error
        return ""


def augment_ground_truth_with_functions(gts: List[Localization], workspace_dir: Path, trace_only: bool = False) -> List[Localization]:
    """
    Augment ground truth localizations with derived function names.
    If a ground truth entry has no function name, derive it from the source code.
    
    Args:
        gts: List of ground truth Localization objects
        workspace_dir: Path to the workspace directory containing src-vul
        trace_only: Whether to output trace logs
    
    Returns:
        List of augmented Localization objects with function names filled in
    """
    augmented_gts = []
    src_vul_dir = workspace_dir / "src-vul"
    
    for gt in gts:
        # If function is already provided and non-empty, keep it
        if gt.function:
            augmented_gts.append(gt)
            continue
        
        # Try to derive function name from source code
        # Ground truth file paths are relative (e.g., "coders/png.c" or "src/njs_regexp.c")
        # We need to find the actual file in src-vul directory
        gt_file_path = src_vul_dir / gt.file
        
        # Strategy 1: Try direct path
        if not gt_file_path.exists():
            # Strategy 2: Try without "src/" prefix if file has it
            if gt.file.startswith("src/"):
                gt_file_path = src_vul_dir / gt.file[4:]
        
        # Strategy 3: Search recursively matching the path structure
        if not gt_file_path.exists():
            filename = os.path.basename(gt.file)
            relative_parts = Path(gt.file).parts  # Get path components (e.g., ("coders", "png.c"))
            
            # Search for files matching this pattern
            possible_files = list(src_vul_dir.glob(f"**/{filename}"))
            if possible_files:
                # Try to find the best match based on path structure
                best_match = None
                for candidate in possible_files:
                    candidate_parts = candidate.relative_to(src_vul_dir).parts
                    
                    # Check if the candidate path ends with the same relative path structure
                    # If the GT path is "coders/png.c", check if candidate ends with "coders/png.c"
                    if len(candidate_parts) >= len(relative_parts):
                        if candidate_parts[-len(relative_parts):] == relative_parts:
                            best_match = candidate
                            break
                
                # Use best match if found, otherwise fall back to first match
                gt_file_path = best_match if best_match else possible_files[0]
        
        if gt_file_path.exists():
            # Use the middle of the span for function lookup
            lookup_line = (gt.old_span.start + gt.old_span.end) // 2
            func_name = derive_function_name(gt_file_path, lookup_line, trace_only)
            
            # Create new Localization with the derived function name
            augmented_gt = Localization(
                task_id=gt.task_id,
                file=gt.file,
                old_span=gt.old_span,
                new_span=gt.new_span,
                function=func_name,
            )
            augmented_gts.append(augmented_gt)
        else:
            # File not found, keep original
            augmented_gts.append(gt)
    
    return augmented_gts
