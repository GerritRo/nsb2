# Contributing to nsb2

## Development Setup

```bash
# Fork on GitHub, then:
git clone https://github.com/<your-username>/nsb2.git
cd nsb2
git remote add upstream https://github.com/GerritRo/nsb2.git

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Install the git hooks that run ruff and the file checks on every commit
pre-commit install

# Verify everything works
pytest
```

nsb2 tries to follow the [ctapipe style guide](https://ctapipe.readthedocs.io/en/stable/developer-guide/style-guide.html).

---

## Workflow (Gitflow)

We use a **Gitflow** model: `main` holds tagged releases, `dev` is the
integration branch. All work happens on short-lived branches off `dev`.

| Branch prefix      | Purpose                 | Target  |
|--------------------|-------------------------|---------|
| `feature/<name>`   | New features            | `dev`   |
| `bugfix/<name>`    | Bug fixes               | `dev`   |
| `hotfix/<x.y.z>`   | Urgent fixes in prod    | `main`  |
| `release/<x.y.z>`  | Release prep            | `main`  |

**Typical contribution flow:**

```bash
git fetch upstream && git checkout -b feature/my-change upstream/dev

# ... develop, commit, test ...

git push origin feature/my-change
# Open a PR targeting dev
```

Keep branches short-lived. Rebase on `upstream/dev` before opening a PR.

---

## Committing with Commitizen

We use [Commitizen](https://commitizen-tools.github.io/commitizen/) to enforce
[Conventional Commits](https://www.conventionalcommits.org/) and generate
changelogs automatically. Use `cz commit` instead of `git commit`:

```bash
cz commit
```

You can also write commit messages manually -- the format is:

```
<type>(<scope>): <description>
```

Breaking changes use `!` after the type: `feat(core)!: require explicit instrument config`

---

## Pull Requests

- **Title** follows Conventional Commits format (it becomes the merge commit message)
- **Target branch** is `dev` (unless it's a hotfix targeting `main`)
- **PR checklist:**
  - [ ] Tests pass (`pytest`)
  - [ ] Lint and formatting pass (`pre-commit run --all-files`)
  - [ ] Types pass (`mypy nsb2`)
  - [ ] New code has tests, and bug fixes have a regression test
  - [ ] New public API has NumPy-style docstrings

---

## Code Quality

### Quick reference

```bash
pre-commit run --all-files               # Everything CI's lint job runs
ruff check nsb2                          # Lint
ruff check --fix nsb2                    # Lint + auto-fix
ruff format nsb2                         # Format
mypy nsb2                                # Type check
pytest                                   # Tests
pytest -m remote_data                    # Tests that download reference data
pytest --cov=nsb2 --cov-report=html      # Coverage report
```

### Where tests live

Tests sit in a `tests/` subdirectory of the module they cover, as in
ctapipe:

```
nsb2/core/tests/test_spectral.py
nsb2/atmosphere/tests/test_single_scattering.py
nsb2/emitter/tests/test_airglow.py
nsb2/instrument/tests/test_bundled_instruments.py
```

Fixtures shared across subpackages go in `nsb2/conftest.py`. Any test that
needs the network must be marked `@pytest.mark.remote_data`, so that the
default `pytest` run stays offline and fast.

### Conventions (as in ctapipe)

- Every public function, class and module carries a
  [NumPy-style docstring](https://numpydoc.readthedocs.io/en/latest/format.html).
- Algorithms cite their source, and the citation is collected in
  `docs/bibliography.rst`. Reference it from the docstring as
  ``[Author2003]_``.
- Use `logging` rather than `print()`; library modules define
  `logger = logging.getLogger(__name__)` at the top of the file. `ruff`
  fails the build on `print()`.
- Functions must not modify their arguments — the pipeline stages are meant
  to be reorderable and parallelisable.
- Use `astropy.units` for any quantity in a public API whose unit could be
  ambiguous.

---

## Documentation

```bash
pip install -e ".[docs]"
cd docs && make html
# Open docs/_build/html/index.html
```

- **Docstrings** use [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html)
- **Example notebooks** go in `docs/examples/` and must be added to `docs/examples/index.rst`

---

## Release Process

Maintainers only. Uses [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`).

1. Create `release/x.y.z` from `dev`
2. `cz bump --changelog` to bump version + generate changelog
3. Run the full test suite **including** `pytest -m remote_data`.
4. Build the docs (`cd docs && make html`); warnings are errors
5. Merge into `main`, tag `vx.y.z`, backmerge into `dev`

---

## Getting Help

- [GitHub Issues](https://github.com/GerritRo/nsb2/issues) for bugs and feature requests
- Email: gerrit.roellinghoff@fau.de

If unsure about a change, open an issue first to discuss the approach.
