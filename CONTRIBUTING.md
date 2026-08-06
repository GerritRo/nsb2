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

nsb2 follows the [ctapipe style guide](https://ctapipe.readthedocs.io/en/stable/developer-guide/style-guide.html)
and is kept co-installable with ctapipe, so that it can be used from a
ctapipe analysis or eventually vendored into it.

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

This walks you through an interactive prompt:

```
? Select the type of change you are committing: (Use arrow keys)
 » fix: A bug fix
   feat: A new feature
   docs: Documentation only changes
   refactor: A code change that neither fixes a bug nor adds a feature
   perf: A code change that improves performance
   test: Adding missing or correcting existing tests
   build: Changes that affect the build system or dependencies
   ci: Changes to CI configuration files and scripts
   chore: Other changes that don't modify src or test files

? What is the scope of this change? (press enter to skip)
  core, emitter, atmosphere, instrument

? Write a short, imperative description of the change:
  > fix airglow model interpolation at high zenith angles

? Provide additional contextual information (press enter to skip):
  > The spline extrapolation produced NaN for zenith > 80 degrees

? Is this a BREAKING CHANGE?  No
? Footer (press enter to skip, e.g. "Closes #42"):
  > Fixes #12
```

Result: `fix(core): fix airglow model interpolation at high zenith angles`

You can also write commit messages manually -- the format is:

```
<type>(<scope>): <description>
```

Breaking changes use `!` after the type: `feat(core)!: require explicit instrument config`

**Why this matters:** `CHANGELOG.md` is generated directly from these commit
messages at release time via `cz bump --changelog`.

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
ruff format nsb2                         # Format (black-compatible, 88 cols)
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

**CI does not run the `remote_data` tests.** They are the only thing that
checks the external reference data is still where we expect it — a retired
CALSPEC revision once broke every star source with a silent 404. Run them
yourself before a release, and whenever you touch an emitter:

```bash
pytest -m remote_data
```

Prefer making code testable offline over adding a new `remote_data` test.
The pattern used throughout is to keep the physics in module-level functions
and let only the data loading be network-bound — see
`nsb2/emitter/moon.py`, where the ROLO albedo model is fully testable
without the solar spectrum download. Downloads themselves can be stubbed
with `nsb2.conftest.stub_download`.

### Reference arrays

`nsb2/core/tests/test_regression.py` pins the offline computation chain to
stored reference values. They detect drift; they do not validate against a
published table. If a change moves them, work out why before regenerating:

```bash
python -m nsb2.core.tests.test_regression
```

### Conventions

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
3. Run the full test suite **including** `pytest -m remote_data` — CI never
   runs those, so this is the only check that the external reference data is
   still reachable
4. Build the docs (`cd docs && make html`); warnings are errors
5. Merge into `main`, tag `vx.y.z`, backmerge into `dev`

---

## Getting Help

- [GitHub Issues](https://github.com/GerritRo/nsb2/issues) for bugs and feature requests
- Email: gerrit.roellinghoff@fau.de

If unsure about a change, open an issue first to discuss the approach.
