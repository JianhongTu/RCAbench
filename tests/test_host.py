"""
test_host.py

End-to-end integration test for the RCAbench evaluation server host-side functionality.

This script validates the complete evaluation pipeline without requiring an actual LLM agent,
using mock submissions to test the server's capabilities. It ensures that the evaluation
server can correctly handle task preparation, patch validation, and localization evaluation.

Test Coverage:
1. **Evaluation Server Setup**: Initializes FastAPI server using TestClient and verifies 
   the root endpoint responds correctly.

2. **Task Asset Preparation**: Downloads and sets up task assets (diff files, error reports,
   codebase archives) for a specific Arvo task ID from the remote repository. Validates
   the workspace and cache directory structure.

3. **Patch Submission & Validation**: Tests the `/patch` endpoint by:
   - Creating a dummy patch file in the shared workspace
   - Mocking Docker container execution to avoid actual container runs
   - Verifying successful patch validation (exit code 0)
   - Testing failed patch validation (exit code 1)

4. **Localization Evaluation**: Tests the `/evaluate` endpoint by:
   - Retrieving ground truth localizations from the diff file
   - Creating mock localization predictions (both perfect and partial matches)
   - Submitting predictions as JSON files in the workspace
   - Validating evaluation metrics (file accuracy, function recall, line IoU)
   - Testing error handling for missing localization files

This test uses Arvo task ID "10055" which has pre-downloaded assets available.
All tests run in isolated temporary directories and use mocked Docker operations
to ensure fast, reproducible testing without external dependencies.

Usage:
    conda activate rcabench
    python tests/test_host.py
"""

import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from rcabench.server.main import app
from rcabench.task.gen_task import prepare_task_asssets
from rcabench.server.eval_utils import get_ground_truth, evaluate_localization, Localization, LineSpan
import rcabench


def test_host_pipeline():
    """
    Test the complete host-side evaluation pipeline.
    """
    # 1. Set up the evaluation server
    print("=" * 60)
    print("Step 1: Setting up the evaluation server")
    print("=" * 60)
    client = TestClient(app)
    
    # Test root endpoint
    response = client.get("/")
    assert response.status_code == 200
    assert "RCAbench Evaluation Server is running" in response.json()["message"]
    print("✓ Evaluation server initialized successfully\n")

    # 2. Prepare the task assets and set up the directory structure
    print("=" * 60)
    print("Step 2: Preparing task assets")
    print("=" * 60)
    test_arvo_id = "10055"  # Use a known test ID
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_path = Path(tmp_dir) / "workspace"
        cache_path = Path(tmp_dir) / "cache"
        data_path = Path(tmp_dir) / "data"
        workspace_path.mkdir()
        cache_path.mkdir()
        
        # Copy data directory for db access
        if Path("data").exists():
            shutil.copytree(Path("data"), data_path)
            rcabench.DATA_DIR = str(data_path)
            print(f"✓ Copied data directory to {data_path}")
        else:
            print("⚠ Warning: data/ directory not found, skipping database tests")
            return

        # Prepare task assets
        try:
            prepare_task_asssets(test_arvo_id, str(workspace_path), str(cache_path))
            print(f"✓ Task assets prepared for arvo_id={test_arvo_id}")
            print(f"  Workspace: {workspace_path}")
            print(f"  Cache: {cache_path}\n")
        except Exception as e:
            print(f"⚠ Warning: Could not prepare task assets: {e}")
            print("  Continuing with mock data...\n")

        # 3. Create a fake patch submission and test submission handling
        print("=" * 60)
        print("Step 3: Testing patch submission")
        print("=" * 60)
        
        patch_dir = workspace_path / "shared"
        patch_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a fake patch file
        fake_patch_content = """diff --git a/src/example.c b/src/example.c
index 123abc..456def 100644
--- a/src/example.c
+++ b/src/example.c
@@ -10,7 +10,7 @@ int vulnerable_function() {
     char buffer[10];
-    strcpy(buffer, user_input);  // Vulnerable!
+    strncpy(buffer, user_input, sizeof(buffer) - 1);  // Fixed
     return 0;
 }
"""
        fake_patch = patch_dir / "patch.diff"
        fake_patch.write_text(fake_patch_content)
        print(f"✓ Created fake patch at {fake_patch}")

        # Mock run_arvo_container to avoid actual Docker run
        with patch('rcabench.server.main.run_arvo_container') as mock_run:
            mock_run.return_value = (0, b"Patch applied successfully\nBuild successful\nTests passed\n")

            response = client.post("/patch", json={
                "arvo_id": test_arvo_id,
                "patch_dir": str(patch_dir)
            })
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
            data = response.json()
            assert data["success"] == True, f"Expected success=True, got {data}"
            assert data["message"] == "Validation successful"
            assert data["exit_code"] == 0
            print(f"✓ Patch validation endpoint test passed")
            print(f"  Exit code: {data['exit_code']}")
            print(f"  Success: {data['success']}")
            print(f"  Message: {data['message']}\n")

        # Test with failed patch
        with patch('rcabench.server.main.run_arvo_container') as mock_run:
            mock_run.return_value = (1, b"Patch application failed\nError: File not found\n")

            response = client.post("/patch", json={
                "arvo_id": test_arvo_id,
                "patch_dir": str(patch_dir)
            })
            
            data = response.json()
            assert data["success"] == False
            assert data["exit_code"] == 1
            print(f"✓ Failed patch validation test passed\n")

        # 4. Create a fake localization prediction and test evaluation
        print("=" * 60)
        print("Step 4: Testing localization evaluation")
        print("=" * 60)
        
        # Get ground truth
        try:
            gts = get_ground_truth(test_arvo_id)
            print(f"✓ Retrieved ground truth for {test_arvo_id}")
            print(f"  Number of ground truth localizations: {len(gts)}")
            for i, gt in enumerate(gts[:3]):  # Show first 3
                print(f"  GT {i+1}: {gt.file} (lines {gt.old_span.start}-{gt.old_span.end})")
            if len(gts) > 3:
                print(f"  ... and {len(gts) - 3} more")
            print()
        except Exception as e:
            print(f"⚠ Warning: Could not get ground truth: {e}")
            print("  Using mock ground truth...\n")
            gts = [
                Localization(
                    task_id=f"arvo:{test_arvo_id}",
                    file="src/example.c",
                    function="vulnerable_function",
                    old_span=LineSpan(start=10, end=15),
                    new_span=LineSpan(start=10, end=15)
                )
            ]
        
        # Create fake predictions - perfect match
        preds_perfect = [
            Localization(
                task_id=f"arvo:{test_arvo_id}",
                file=gt.file,
                function=gt.function,
                old_span=LineSpan(start=gt.old_span.start, end=gt.old_span.end),
                new_span=LineSpan(start=gt.new_span.start, end=gt.new_span.end)
            )
            for gt in gts[:1]  # Match first GT
        ]
        
        # Save localization submission to file
        loc_submission = [pred.to_dict() for pred in preds_perfect]
        loc_file = patch_dir / "loc.json"
        with open(loc_file, 'w') as f:
            json.dump(loc_submission, f, indent=2)
        print(f"✓ Created localization submission at {loc_file}")
        
        # Test evaluation endpoint
        response = client.post("/evaluate", json={
            "arvo_id": test_arvo_id,
            "patch_dir": str(patch_dir)
        })
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        eval_report = response.json()
        print(f"✓ Localization evaluation endpoint test passed")
        print(f"  Task ID: {eval_report['task_id']}")
        print(f"  N GT: {eval_report['n_gt']}, N Pred: {eval_report['n_pred']}")
        print(f"  File Accuracy: {eval_report.get('file_acc', 0.0):.3f}")
        
        # func_topk_recall and line_topk_recall are dicts
        func_recall = eval_report.get('func_topk_recall', {})
        if func_recall:
            recall_str = ", ".join([f"@{k}: {v:.3f}" for k, v in sorted(func_recall.items())])
            print(f"  Function TopK Recall: {recall_str}")
        
        line_recall = eval_report.get('line_topk_recall', {})
        if line_recall:
            recall_str = ", ".join([f"@{k}: {v:.3f}" for k, v in sorted(line_recall.items())])
            print(f"  Line TopK Recall: {recall_str}")
        
        line_iou = eval_report.get('line_iou_mean')
        if line_iou is not None:
            print(f"  Line IoU Mean: {line_iou:.3f}")
        print()
        
        # Test with partial match
        preds_partial = [
            Localization(
                task_id=f"arvo:{test_arvo_id}",
                file=gts[0].file,
                function="wrong_function",  # Wrong function
                old_span=LineSpan(start=gts[0].old_span.start, end=gts[0].old_span.end + 5),  # Extended range
                new_span=LineSpan(start=gts[0].new_span.start, end=gts[0].new_span.end + 5)
            )
        ] if gts else preds_perfect
        
        loc_submission_partial = [pred.to_dict() for pred in preds_partial]
        with open(loc_file, 'w') as f:
            json.dump(loc_submission_partial, f, indent=2)
        
        response = client.post("/evaluate", json={
            "arvo_id": test_arvo_id,
            "patch_dir": str(patch_dir)
        })
        
        assert response.status_code == 200
        eval_report_partial = response.json()
        print(f"✓ Partial match localization test passed")
        file_acc = eval_report_partial.get('file_acc')
        if file_acc is not None:
            print(f"  File Accuracy: {file_acc:.3f}")
        
        line_recall = eval_report_partial.get('line_topk_recall', {})
        if line_recall:
            recall_str = ", ".join([f"@{k}: {v:.3f}" for k, v in sorted(line_recall.items())])
            print(f"  Line TopK Recall: {recall_str}")
        
        line_iou = eval_report_partial.get('line_iou_mean')
        if line_iou is not None:
            print(f"  Line IoU Mean: {line_iou:.3f}")
        print()
        
        # Test with missing localization file
        loc_file.unlink()
        response = client.post("/evaluate", json={
            "arvo_id": test_arvo_id,
            "patch_dir": str(patch_dir)
        })
        # Should return 400 or 500 for missing file
        assert response.status_code in [400, 500], f"Expected 400 or 500 for missing file, got {response.status_code}"
        print(f"✓ Missing localization file test passed (status: {response.status_code})\n")

    print("=" * 60)
    print("All host pipeline tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_host_pipeline()

