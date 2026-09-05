# Releasing

1. Verify that CI is green and the version in `pyproject.toml` is final.
2. Build and validate locally with `python -m build` and
   `python -m twine check dist/*`.
3. Optionally run the manual TestPyPI workflow and install the uploaded version in a
   clean environment.
4. Create a GitHub release whose tag is `v` followed by the package version.
5. The release workflow builds fresh artifacts and publishes them to PyPI through
   Trusted Publishing.
6. Install the release from PyPI in a clean environment and run the README example.

Before the first upload, configure this repository as a Trusted Publisher separately
on TestPyPI and PyPI. Use GitHub environments named `testpypi` and `pypi`; no API token
is stored in GitHub.
