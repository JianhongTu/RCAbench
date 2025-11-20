"""
Quick test to see if Docker is accessible from Python
"""

import docker

try:
    client = docker.from_env()
    print("✓ Docker client created successfully")
    
    # Try to get Docker version
    version = client.version()
    print(f"✓ Docker version: {version['Version']}")
    
    # Try to list images
    images = client.images.list()
    print(f"✓ Found {len(images)} local Docker images")
    
    # Try to pull a test image
    print("\nTrying to pull n132/arvo:10055-vul...")
    image = client.images.pull("n132/arvo:10055-vul")
    print(f"✓ Successfully pulled image: {image.tags}")
    
except docker.errors.DockerException as e:
    print(f"✗ Docker error: {e}")
    print("\nPossible issues:")
    print("  1. Docker daemon not running")
    print("  2. Permission issues (need to add user to docker group)")
    print("  3. Docker socket not accessible")
except Exception as e:
    print(f"✗ Unexpected error: {e}")

