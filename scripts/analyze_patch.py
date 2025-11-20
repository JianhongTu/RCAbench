"""
analyze_patch.py

This script analyzes a single patch file to extract metadata:
- How many files are changed
- How many lines are added/removed
- What types of files (code vs docs/config)
- Function names that are modified
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple


def parse_patch_file(patch_content: str) -> Dict:
    """
    Parse a unified diff patch file and extract metadata.
    
    A unified diff looks like:
        diff --git a/file1.c b/file1.c
        --- a/file1.c
        +++ b/file1.c
        @@ -100,5 +100,6 @@ function_name
        context line
        -deleted line
        +added line
    
    Returns:
        Dictionary with patch metadata
    """
    
    # This regex finds file headers in the diff
    # It captures the filename from "+++ b/filename" (Git format)
    # or "+++ b/filename\tTimestamp" (Mercurial format)
    # The pattern captures everything after "b/" until tab or newline
    file_pattern = re.compile(r'^\+\+\+ b/([^\t\n]+)', re.MULTILINE)
    
    # This regex finds hunks (sections of changes)
    # Format: @@ -old_start,old_count +new_start,new_count @@ optional_context
    hunk_pattern = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$', re.MULTILINE)
    
    # Find all files modified
    files = file_pattern.findall(patch_content)
    
    # Count lines added/removed
    lines = patch_content.split('\n')
    added_lines = 0
    removed_lines = 0
    
    for line in lines:
        if line.startswith('+') and not line.startswith('+++'):
            added_lines += 1
        elif line.startswith('-') and not line.startswith('---'):
            removed_lines += 1
    
    # Extract function context from hunks (if present)
    functions = []
    for match in hunk_pattern.finditer(patch_content):
        context = match.group(5).strip()  # The part after @@
        if context:
            functions.append(context)
    
    return {
        'files': files,
        'num_files': len(files),
        'lines_added': added_lines,
        'lines_removed': removed_lines,
        'total_changes': added_lines + removed_lines,
        'functions': functions,
        'num_functions': len(functions)
    }


def categorize_file_type(filename: str) -> str:
    """
    Categorize a file into: code, test, doc, or config.
    
    This helps us understand what kind of changes are in the patch.
    """
    filename_lower = filename.lower()
    
    # Code files - these are what we care about for vulnerability analysis
    code_extensions = {'.c', '.cc', '.cpp', '.cxx', '.h', '.hpp', '.hh', 
                       '.rs', '.go', '.py', '.java', '.js', '.ts'}
    
    # Test files - these don't affect the actual vulnerability
    if '/test/' in filename_lower or filename_lower.startswith('test'):
        return 'test'
    
    # Documentation - not relevant for vulnerability localization
    doc_extensions = {'.md', '.txt', '.rst', '.html', '.htm'}
    if any(filename_lower.endswith(ext) for ext in doc_extensions):
        return 'doc'
    
    # Config/build files - usually not relevant
    config_files = {'changelog', 'makefile', 'cmake', '.gitignore', '.yml', '.yaml', 'dockerfile'}
    if any(keyword in filename_lower for keyword in config_files):
        return 'config'
    
    # Check if it's a code file by extension
    ext = Path(filename).suffix.lower()
    if ext in code_extensions:
        return 'code'
    
    return 'other'


def analyze_patch(patch_path: str) -> Dict:
    """
    Main function to analyze a patch file.
    
    Args:
        patch_path: Path to the .diff file
        
    Returns:
        Dictionary with complete patch analysis
    """
    with open(patch_path, 'r', encoding='utf-8', errors='ignore') as f:
        patch_content = f.read()
    
    # Get basic patch metadata
    metadata = parse_patch_file(patch_content)
    
    # Categorize each file
    file_categories = {}
    for filename in metadata['files']:
        file_categories[filename] = categorize_file_type(filename)
    
    # Count files by category
    category_counts = {}
    for category in file_categories.values():
        category_counts[category] = category_counts.get(category, 0) + 1
    
    # Number of actual code files changed (this is what matters!)
    num_code_files = category_counts.get('code', 0)
    
    metadata.update({
        'file_categories': file_categories,
        'category_counts': category_counts,
        'num_code_files': num_code_files
    })
    
    return metadata


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_patch.py <patch_file>")
        sys.exit(1)
    
    patch_path = sys.argv[1]
    result = analyze_patch(patch_path)
    
    print("\n" + "="*60)
    print("PATCH ANALYSIS")
    print("="*60)
    print(f"Total files changed: {result['num_files']}")
    print(f"Code files changed: {result['num_code_files']}")
    print(f"Lines added: {result['lines_added']}")
    print(f"Lines removed: {result['lines_removed']}")
    print(f"Total changes: {result['total_changes']}")
    
    print(f"\nFile breakdown:")
    for category, count in result['category_counts'].items():
        print(f"  {category}: {count}")
    
    print(f"\nFiles:")
    for filename, category in result['file_categories'].items():
        print(f"  [{category}] {filename}")
    
    if result['functions']:
        print(f"\nFunction contexts found:")
        for func in result['functions'][:5]:  # Show first 5
            print(f"  {func}")

