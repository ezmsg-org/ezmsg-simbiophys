# ezmsg-simbiophys

[ezmsg](https://www.ezmsg.org) namespace package for simulating biophysical signals such as ECG, EEG, and intracranial recordings.

## Installation

Install from PyPI:

```bash
pip install ezmsg-simbiophys
```

Or install the latest development version:

```bash
pip install git+https://github.com/ezmsg-org/ezmsg-simbiophys@dev
```

## Dependencies

- `ezmsg`
- `ezmsg-baseproc`
- `ezmsg-sigproc`
- `numpy`


## Development

We use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for development.

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if not already installed.
2. Fork this repository and clone your fork locally.
3. Open a terminal and `cd` to the cloned folder.
4. Run `uv sync` to create a `.venv` and install dependencies.
5. (Optional) Install pre-commit hooks: `uv run pre-commit install`
6. After making changes, run the test suite: `uv run pytest tests`

## License

MIT License - see [LICENSE](LICENSE) for details.
