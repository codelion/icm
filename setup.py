from setuptools import setup, find_packages

setup(
    name="icm",
    version="0.0.1",
    description="Internal Coherence Maximization - Unsupervised Elicitation of Language Models",
    author="codelion",
    author_email="codelion@okyasoft.com",
    url="https://github.com/codelion/icm",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.20.0",
        "tqdm>=4.65.0",
        "huggingface_hub>=0.16.0",
        "accelerate",
        "openai>=1.0.0",  # For potential Claude API integration
        "anthropic>=0.7.0",  # For Claude models
        "pydantic>=2.0.0",  # For data validation
        "scipy>=1.9.0",  # For statistical functions
    ],
    entry_points={
        "console_scripts": [
            "icm=icm.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
