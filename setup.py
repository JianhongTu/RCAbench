import setuptools

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="rcabench",
    version="0.1.0",
    author="RCAbench Team",
    description="A cybersecurity benchmark system for root-cause analysis using LLM agents",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/rcabench",  # Replace with actual repo URL
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",  # Adjust if different
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
