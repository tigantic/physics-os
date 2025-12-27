## Description

<!-- Describe your changes in detail -->

## Related Issue

<!-- Link to the issue this PR addresses, e.g., "Fixes #123" or "Relates to #456" -->

## Type of Change

<!-- Mark the relevant option with an [x] -->

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🧪 Test addition or modification
- [ ] ♻️ Refactoring (no functional changes)
- [ ] 🔧 Configuration change
- [ ] 🔒 Security fix

## Checklist

<!-- Mark completed items with an [x] -->

### Required for All PRs

- [ ] I have read the [CONTRIBUTING.md](../CONTRIBUTING.md) guide
- [ ] My code follows the project's code style (ran `ruff check .` and `ruff format .`)
- [ ] I have added/updated tests that prove my fix is effective or that my feature works
- [ ] All existing tests pass locally (`pytest tests/ -v`)
- [ ] My changes generate no new warnings

### For Code Changes

- [ ] I have added type hints to new/modified functions
- [ ] I have updated docstrings for new/modified functions
- [ ] New functions have at least one example in their docstring

### For CFD/Physics Changes

- [ ] I have verified physical conservation properties
- [ ] I have added/updated proof tests in `proofs/`
- [ ] Results match reference solutions within tolerance

### For Breaking Changes

- [ ] I have updated the CHANGELOG.md
- [ ] I have updated relevant documentation
- [ ] I have provided a migration guide

## Test Results

<!-- Include relevant test output or screenshots -->

```
# Paste test output here
pytest tests/ -v
```

## Additional Notes

<!-- Any additional information that reviewers should know -->
