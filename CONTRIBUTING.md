# Contributing to nsb2

## Development Setup

```bash
# Fork on GitHub, then:
git clone https://github.com/<your-username>/nsb2.git
cd nsb2
git remote add upstream https://github.com/GerritRo/nsb2.git

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Verify everything works
pytest
```

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
  - [ ] Lint passes (`ruff check nsb2`)
  - [ ] Types pass (`mypy nsb2`)
  - [ ] New code has tests

---

## Code Quality

### Quick reference

```bash
ruff check nsb2                          # Lint
ruff check --fix nsb2                    # Lint + auto-fix
ruff format nsb2                         # Format
mypy nsb2                                # Type check
pytest                                   # Tests
pytest --cov=nsb2 --cov-report=html      # Coverage report
```

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
3. Run full test suite + doc build
4. Merge into `main`, tag `vx.y.z`, backmerge into `dev`

---

## Getting Help

- [GitHub Issues](https://github.com/GerritRo/nsb2/issues) for bugs and feature requests
- Email: gerrit.roellinghoff@fau.de

If unsure about a change, open an issue first to discuss the approach.