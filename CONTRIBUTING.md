# Contributing

Install the project and development tools in an isolated environment:

```bash
python -m pip install -e ".[dev]"
```

Run the local checks before opening a pull request:

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src
```

The core runtime must remain independent of ML frameworks: do not add mandatory
PyTorch, Transformers, PEFT, TRL, model-download, network, or GPU requirements.
Arbitrary dataset objects referenced by `LearningEpisode` must not be serialized.
Public API additions should be deliberate and accompanied by tests and documentation.
Tests must run without network access, GPUs, or model downloads.
