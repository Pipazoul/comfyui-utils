# ComfyUI Utils Development Guide

## Commands
- **Setup Development Environment**: `pip install -e .[dev]`
- **Install Pre-commit Hooks**: `pre-commit install`
- **Run Linting**: `ruff check .`
- **Testing**: Use pytest for unit testing (`pytest tests/`)
- **Single Test**: `pytest tests/test_file.py::test_function -v`

## Code Style Guidelines
- **Python Version**: Follow ComfyUI's standard (Python 3.x)
- **Formatting**: Use Ruff for code formatting
- **Naming**:
  - Classes: PascalCase (ex: `TimesTwo`)
  - Functions: camelCase (ex: `funcTimesTwo`)
  - Variables: snake_case or camelCase
  - Constants: UPPERCASE
- **Node Structure**:
  - Define `INPUT_TYPES`, `RETURN_TYPES`, `RETURN_NAMES` class variables
  - Use `FUNCTION` to specify the processing method
  - Set `CATEGORY` to identify node group
- **Error Handling**: Use appropriate try/except blocks with meaningful error messages
- **Documentation**: Document classes with docstrings explaining purpose, inputs, and outputs

## Imports & Dependencies
- Keep dependencies minimal and compatible with ComfyUI
- Import third-party libraries only when necessary
- Check for dependencies in the main ComfyUI installation before adding new ones

## Git Workflow
- Make atomic commits with clear messages
- Reference issue numbers in commit messages when applicable
- Create pull requests with clear descriptions of changes