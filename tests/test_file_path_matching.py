#!/usr/bin/env python3
"""
Test file path matching logic to ensure it handles different root directory structures.
"""

import sys
sys.path.append('src')

from rcabench.server.eval_utils import _normalize_file_path, _files_match


def test_normalize_file_path():
    """Test path normalization."""
    test_cases = [
        # (input, expected_output)
        ("graphicsmagick/magick/render.c", "magick/render.c"),
        ("src-vul/graphicsmagick/magick/render.c", "magick/render.c"),
        ("src/graphicsmagick/magick/render.c", "magick/render.c"),
        ("magick/render.c", "magick/render.c"),
        ("render.c", "render.c"),
        ("/workspace/src-vul/magick/render.c", "magick/render.c"),
        ("workspace/graphicsmagick/magick/render.c", "magick/render.c"),
        ("", ""),
        ("/", ""),
    ]
    
    print("Testing _normalize_file_path:")
    for input_path, expected in test_cases:
        result = _normalize_file_path(input_path)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{input_path}' -> '{result}' (expected: '{expected}')")
        assert result == expected, f"Failed: '{input_path}' -> '{result}', expected '{expected}'"
    print()


def test_files_match():
    """Test file matching with various path combinations."""
    # Pairs that SHOULD match (same file, different root structures)
    should_match = [
        ("graphicsmagick/magick/render.c", "magick/render.c"),
        ("src-vul/magick/render.c", "magick/render.c"),
        ("src/graphicsmagick/magick/render.c", "magick/render.c"),
        ("magick/render.c", "render.c"),
        ("render.c", "magick/render.c"),
        ("graphicsmagick/magick/render.c", "src-vul/magick/render.c"),
        ("workspace/src-vul/magick/render.c", "magick/render.c"),
        ("utility.c", "magick/utility.c"),
        ("magick/utility.c", "utility.c"),
        ("graphicsmagick/magick/utility.c", "magick/utility.c"),
        # Same paths
        ("magick/render.c", "magick/render.c"),
        ("render.c", "render.c"),
    ]
    
    # Pairs that SHOULD NOT match (different files)
    should_not_match = [
        ("magick/render.c", "magick/write.c"),
        ("render.c", "write.c"),
        ("magick/render.c", "util/render.c"),
        ("src/file1.c", "src/file2.c"),
    ]
    
    print("Testing _files_match (should match):")
    for path1, path2 in should_match:
        result = _files_match(path1, path2)
        status = "✓" if result else "✗"
        print(f"  {status} '{path1}' <-> '{path2}': {result}")
        assert result, f"Should match: '{path1}' <-> '{path2}'"
    print()
    
    print("Testing _files_match (should NOT match):")
    for path1, path2 in should_not_match:
        result = _files_match(path1, path2)
        status = "✓" if not result else "✗"
        print(f"  {status} '{path1}' <-> '{path2}': {result}")
        assert not result, f"Should NOT match: '{path1}' <-> '{path2}'"
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("File Path Matching Tests")
    print("=" * 60)
    print()
    
    test_normalize_file_path()
    test_files_match()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


