# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GL RL Model is a reinforcement learning model project for General Ledger systems. It implements reasoning models from scratch with a focus on financial data processing and analysis.

## Development Commands

### Setup & Installation
- `uv venv`: Create a new virtual environment
- `source .venv/bin/activate`: Activate the virtual environment
- `uv pip install -e .`: Install the package in editable mode
- `uv pip install pytest ruff pytest-ruff build twine`: Install development dependencies

### Testing & Quality
- `python -m pytest`: Run all tests
- `python -m pytest tests/test_specific.py`: Run a specific test file
- `python -m pytest -v`: Run tests with verbose output
- `python -m ruff check .`: Run linting checks
- `python -m ruff format .`: Format code automatically

### Build & Distribution
- `python -m build`: Build distribution packages
- `python -m twine upload dist/*`: Upload to PyPI (requires credentials)

## Project Structure

The project is organized as a Python package with the following structure:
- `gl_rl_model/`: Main package directory containing the core implementation
- `pyproject.toml`: Project configuration and dependencies
- Tests should be placed in a `tests/` directory when created

## Dependencies & Architecture

### Core Dependencies
- **PyTorch (v2.7.1)**: Deep learning framework for model implementation
- **Tokenizers**: For text tokenization and preprocessing
- **SymPy**: Symbolic mathematics for verification tasks
- **TRL**: For Group Relative Policy Optimization (GRPO) implementation
- **PEFT**: Parameter-Efficient Fine-Tuning with LoRA support
- **Datasets**: HuggingFace datasets for data loading
- **Accelerate**: Distributed training support

### Development Environment
- Python 3.10+ required
- Uses `uv` for dependency management
- JupyterLab included for interactive development and experimentation

## Code Style Guidelines

- Follow PEP 8 conventions for Python code
- Use type hints for function signatures
- Document classes and functions with docstrings
- Keep module imports organized: standard library, third-party, local
- Use meaningful variable names that reflect financial/RL domain context