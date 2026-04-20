# Contributing Guidelines

We welcome contributions! Please follow these steps to ensure a smooth workflow.

## 1. Fork the Repository
Click the **Fork** button on GitHub to create your own copy of the project.

## 2. Clone Your Fork
```bash
git clone https://github.com/<your‑username>/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

## 3. Create a Feature Branch
```bash
git checkout -b <feature‑name>
```
Use a descriptive name, e.g., `add‑new‑model‑benchmark`.

## 4. Development Guidelines
- **Code Style**: Follow PEP 8 for Python and the existing ESLint configuration for the Next.js frontend.
- **Testing**: Add unit tests for any new Python modules under `ml/` and ensure they pass with `pytest`.
- **Documentation**: Update the relevant markdown files (`README.md`, `API.md`, etc.) when adding new functionality.
- **Commit Messages**: Use clear, imperative style. Example: `Add SMOTE hyper‑parameter to training script`.

## 5. Submit a Pull Request
```bash
git push origin <feature‑name>
```
Then open a PR on GitHub, linking any related issue.

## 6. Review Process
- CI will run linting and tests automatically.
- A maintainer will review the PR, suggest changes if needed, and merge once approved.

## 7. Code of Conduct
Please adhere to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

Thank you for helping improve this project!
