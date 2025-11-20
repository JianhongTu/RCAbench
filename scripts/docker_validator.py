"""
docker_validator.py

Tests if patches actually compile and fix vulnerabilities using Docker containers.
This is Stage 2 validation - beyond just checking if patches apply.
"""

import docker
from pathlib import Path
from typing import Dict, Tuple
import tempfile
import shutil


def test_patch_with_docker(task_id: str, patch_path: str, timeout: int = 300) -> Dict:
    """
    Test a patch using Docker containers.
    
    This function:
    1. Checks if Docker image exists (n132/arvo:TASKID-vul)
    2. Applies the patch in the container
    3. Attempts to recompile the code
    4. Runs the fuzzer POC to verify the fix
    
    Args:
        task_id: The ARVO/HuggingFace task ID
        patch_path: Path to the patch file
        timeout: Timeout in seconds for container execution
        
    Returns:
        Dictionary with test results
    """
    
    result = {
        'task_id': task_id,
        'docker_image_exists': False,
        'patch_applied': False,
        'patch_exit_code': None,
        'code_compiled': False,
        'compile_exit_code': None,
        'vulnerability_fixed': False,
        'fuzzer_exit_code': None,
        'docker_output': None,
        'error_message': None
    }
    
    try:
        client = docker.from_env()
        
        # Step 1: Check if Docker image exists
        image_name = f"n132/arvo:{task_id}-vul"
        print(f"  Checking Docker image: {image_name}")
        
        try:
            # Try to pull the image
            client.images.pull(image_name)
            result['docker_image_exists'] = True
            print(f"    ✓ Docker image exists")
        except docker.errors.ImageNotFound:
            result['docker_image_exists'] = False
            result['error_message'] = f"Docker image {image_name} not found"
            print(f"    ✗ Docker image not found")
            return result
        except Exception as e:
            result['error_message'] = f"Error pulling image: {e}"
            print(f"    ✗ Error pulling image: {e}")
            return result
        
        # Step 2: Prepare patch directory
        # Docker needs the patch in a directory it can mount
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_patch = Path(temp_dir) / f"{task_id}_patch.diff"
            shutil.copy(patch_path, temp_patch)
            
            # Step 3: Run container with patch
            print(f"  Running Docker container to test patch...")
            
            # Command to run inside container:
            # Run each step separately and capture exit codes
            # Output format: "PATCH_EXIT:X COMPILE_EXIT:Y FUZZER_EXIT:Z"
            cmd = [
                "/bin/sh", "-c",
                f"echo '=== Step 1: Applying patch ===' && "
                f"patch -p1 --force < /tmp/patch/{task_id}_patch.diff; "
                f"PATCH_EXIT=$?; "
                f"echo \"PATCH_EXIT:$PATCH_EXIT\"; "
                f"echo '=== Step 2: Compiling code ===' && "
                f"arvo compile; "
                f"COMPILE_EXIT=$?; "
                f"echo \"COMPILE_EXIT:$COMPILE_EXIT\"; "
                f"echo '=== Step 3: Running fuzzer POC ===' && "
                f"timeout {timeout} arvo; "
                f"FUZZER_EXIT=$?; "
                f"echo \"FUZZER_EXIT:$FUZZER_EXIT\"; "
                f"echo \"ALL_DONE\""
            ]
            
            try:
                container = client.containers.run(
                    image=image_name,
                    command=cmd,
                    volumes={
                        str(Path(temp_dir).absolute()): {"bind": "/tmp/patch", "mode": "ro"}
                    },
                    detach=True,
                    remove=False  # Don't auto-remove so we can inspect
                )
                
                # Stream logs and collect them
                print(f"    Container started, running tests...")
                logs = []
                try:
                    for line in container.logs(stream=True, follow=True):
                        line_str = line.decode('utf-8', errors='ignore').strip()
                        logs.append(line_str)
                        # Only print critical lines (steps and exit codes)
                        if any(keyword in line_str for keyword in ['=== Step', 'EXIT:', 'ALL_DONE']):
                            print(f"      {line_str}")
                    
                    # Wait for container to finish (with longer timeout)
                    container.wait(timeout=timeout + 60)
                except Exception as e:
                    print(f"    Warning during log streaming: {e}")
                
                # Combine all logs
                output = '\n'.join(logs)
                result['docker_output'] = output
                
                # Save full output to file for debugging (outside temp_dir so it persists)
                log_dir = Path("./tmp/docker_logs")
                log_dir.mkdir(parents=True, exist_ok=True)
                output_file = log_dir / f"{task_id}_docker_output.log"
                with open(output_file, 'w') as f:
                    f.write(output)
                print(f"    (Full logs saved to: {output_file})")
                
                # Parse exit codes from output
                import re
                
                patch_match = re.search(r'PATCH_EXIT:(\d+)', output)
                compile_match = re.search(r'COMPILE_EXIT:(\d+)', output)
                fuzzer_match = re.search(r'FUZZER_EXIT:(\d+)', output)
                
                if patch_match:
                    result['patch_exit_code'] = int(patch_match.group(1))
                    result['patch_applied'] = (result['patch_exit_code'] == 0)
                    if result['patch_applied']:
                        print(f"    ✓ Step 1: Patch applied successfully")
                    else:
                        print(f"    ✗ Step 1: Patch failed (exit code: {result['patch_exit_code']})")
                
                if compile_match:
                    result['compile_exit_code'] = int(compile_match.group(1))
                    result['code_compiled'] = (result['compile_exit_code'] == 0)
                    if result['code_compiled']:
                        print(f"    ✓ Step 2: Code compiled successfully")
                    else:
                        print(f"    ✗ Step 2: Compilation failed (exit code: {result['compile_exit_code']})")
                
                if fuzzer_match:
                    result['fuzzer_exit_code'] = int(fuzzer_match.group(1))
                    # Exit code 0 means no crash (vulnerability fixed!)
                    # Non-zero means crash still happens
                    result['vulnerability_fixed'] = (result['fuzzer_exit_code'] == 0)
                    if result['vulnerability_fixed']:
                        print(f"    ✓ Step 3: Fuzzer POC passed - vulnerability FIXED!")
                    else:
                        print(f"    ✗ Step 3: Fuzzer POC failed - vulnerability still exists (exit code: {result['fuzzer_exit_code']})")
                
                # Overall assessment
                if result['patch_applied'] and result['code_compiled'] and result['vulnerability_fixed']:
                    print(f"    🎉 FULL SUCCESS: Patch works perfectly!")
                elif result['patch_applied'] and result['code_compiled']:
                    print(f"    ⚠️  PARTIAL: Patch compiles but doesn't fix the vulnerability")
                elif result['patch_applied']:
                    print(f"    ⚠️  PARTIAL: Patch applies but doesn't compile")
                else:
                    print(f"    ❌ FAILED: Patch doesn't even apply")
                
                # Clean up container
                container.remove(force=True)
                
            except docker.errors.ContainerError as e:
                result['error_message'] = f"Container error: {e}"
                print(f"    ✗ Container error: {e}")
            except Exception as e:
                result['error_message'] = f"Execution error: {e}"
                print(f"    ✗ Execution error: {e}")
    
    except Exception as e:
        result['error_message'] = f"Docker client error: {e}"
        print(f"  ✗ Docker client error: {e}")
    
    return result


def check_docker_image_exists(task_id: str) -> bool:
    """
    Quick check if a Docker image exists for a task.
    
    Args:
        task_id: The task ID
        
    Returns:
        True if image exists, False otherwise
    """
    try:
        client = docker.from_env()
        image_name = f"n132/arvo:{task_id}-vul"
        
        # Try to inspect the image (faster than pulling)
        try:
            client.images.get(image_name)
            return True
        except docker.errors.ImageNotFound:
            # Try pulling it
            try:
                client.images.pull(image_name)
                return True
            except:
                return False
    except:
        return False


if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python docker_validator.py <task_id> [patch_path]")
        sys.exit(1)
    
    task_id = sys.argv[1]
    patch_path = sys.argv[2] if len(sys.argv) > 2 else f"./tmp/validation/{task_id}_patch.diff"
    
    print(f"Testing Docker validation for task {task_id}")
    print("="*60)
    
    # First check if image exists
    print("Checking if Docker image exists...")
    if check_docker_image_exists(task_id):
        print(f"✓ Docker image exists for task {task_id}")
        
        # Run full validation
        result = test_patch_with_docker(task_id, patch_path)
        
        print("\n" + "="*60)
        print("Results:")
        # Print result without the massive docker_output field
        result_summary = {k: v for k, v in result.items() if k != 'docker_output'}
        print(json.dumps(result_summary, indent=2))
        
        if result.get('docker_output'):
            print(f"\n(Full Docker output: {len(result['docker_output'])} characters)")
            print(f"Saved to: tmp/docker_logs/{task_id}_docker_output.log")
    else:
        print(f"✗ No Docker image available for task {task_id}")
        print("This task cannot be validated with Docker")

