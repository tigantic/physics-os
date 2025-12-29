# Commit Checklist - Quick Reference

## ✅ Before Every Commit

```bash
# 1. Verify changes compile/run
cargo build --release  # If Rust changed
python quick_test.py   # If Python changed

# 2. Check git status
git status

# 3. Stage specific files (not `git add .`)
git add [specific-files]

# 4. Write descriptive commit
git commit -m "type(scope): description"
```

## ✅ Daily Workflow

**Start of Day**:
```bash
cd /home/brad/TiganticLabz/Main_Projects/'Project HyperTensor'
git status  # Check for uncommitted work
git pull origin main  # Sync with remote (if collaborating)
```

**End of Development Session**:
```bash
# Commit completed work
git add [files]
git commit -m "feat(scope): what you did"

# Push when stable (not every commit)
git push origin main
```

## ✅ Phase Milestone Checklist

**Phase Completion (e.g., Glass Cockpit Phase 3)**:

- [ ] All deliverables complete (check phase doc)
- [ ] Tests passing (`python quick_test.py`)
- [ ] GPU validation (`python test_100k_stress.py` if relevant)
- [ ] Create attestation: `PHASE[N]_[NAME]_ATTESTATION.json`
- [ ] Update `CHANGELOG.md`
- [ ] Update `ROADMAP.md`
- [ ] Create phase completion document (e.g., `PHASE_3_COMPLETE.md`)
- [ ] Commit all changes
- [ ] Tag release: `git tag -a v0.[N].0 -m "Phase [N] complete"`
- [ ] Push: `git push origin main --tags`

## ✅ Commit Message Template

```
type(scope): Brief description (50 chars max)

Longer explanation of what changed and why (wrap at 72 chars).
Include motivation and contrast with previous behavior.

Deliverables:
- Item 1 completed
- Item 2 validated

Testing:
- Test results: [X/X passed]
- Validation: [evidence file path]

Constitutional Compliance:
- Article [X]: [requirement met]

Related: #[issue], PHASE_[N]_COMPLETE.md
```

## ✅ Commit Types

| Type | When to Use | Example |
|------|------------|---------|
| `feat` | New feature | `feat(glass-cockpit): Add 3D tensor glyphs` |
| `fix` | Bug fix | `fix(renderer): Correct depth buffer precision` |
| `docs` | Documentation | `docs(roadmap): Update Phase 3 timeline` |
| `test` | Tests only | `test(gpu): Add 4K validation suite` |
| `refactor` | Code restructure | `refactor(bridge): Simplify telemetry struct` |
| `perf` | Performance | `perf(qtt): Optimize decompression kernel` |
| `chore` | Maintenance | `chore(deps): Update wgpu to 0.20` |

## ✅ Scopes (Common)

- `glass-cockpit` - Rust rendering frontend
- `sovereign` - Python physics engine
- `bridge` - RAM bridge integration
- `gpu` - GPU acceleration
- `qtt` - QTT tensor operations
- `mpo` - MPO solver
- `docs` - Documentation
- `tests` - Test infrastructure

## ✅ Push Checklist

**Before Pushing**:
- [ ] All commits have meaningful messages
- [ ] Tests pass (`python quick_test.py`)
- [ ] No debug code or TODOs in committed files
- [ ] Sensitive data removed (API keys, credentials)
- [ ] Large binaries justified or use Git LFS

**Push Command**:
```bash
git push origin main
```

**With Tags** (for releases):
```bash
git push origin main --tags
```

## ✅ Emergency Rollback

**Undo Last Commit (Keep Changes)**:
```bash
git reset --soft HEAD~1
```

**Undo Last Commit (Discard Changes)**:
```bash
git reset --hard HEAD~1
# WARNING: Destroys uncommitted work!
```

**Revert Pushed Commit**:
```bash
git revert <commit-hash>
git push origin main
```

## ✅ Quick Status Check

```bash
# What's changed?
git status

# Recent commits
git log --oneline -10

# Uncommitted diff
git diff

# Staged diff
git diff --cached

# Who changed what?
git blame <file>

# Branch status
git branch -vv
```

## ✅ Common Tasks

**Add New Files**:
```bash
git add path/to/new_file.rs
git commit -m "feat(scope): Add new_file"
```

**Remove Tracked File**:
```bash
git rm path/to/old_file.py
git commit -m "chore: Remove obsolete old_file"
```

**Rename File**:
```bash
git mv old_name.py new_name.py
git commit -m "refactor: Rename old_name to new_name"
```

**Update .gitignore**:
```bash
echo "new_ignore_pattern/" >> .gitignore
git add .gitignore
git commit -m "chore: Update .gitignore"
```

## ✅ Constitutional Compliance

Per CONSTITUTION.md Article VI, every commit should:

- ✅ Use clear, descriptive messages (not "WIP" or "fix stuff")
- ✅ Reference evidence/validation when applicable
- ✅ Update CHANGELOG.md for user-facing changes
- ✅ Update DECISION_LOG.md for architecture changes
- ✅ Include test results for features/fixes
- ✅ Link to attestations for phase completions

## ✅ Pre-Push Authentication

**If push requires authentication**:

Option 1: **Personal Access Token** (HTTPS)
```bash
# GitHub prompts for username/password
# Use token as password (not actual password)
```

Option 2: **SSH Key** (Recommended)
```bash
# Update remote to SSH
git remote set-url origin git@github.com:tigantic/HyperTensor.git

# Push with SSH
git push origin main
```

## ✅ Troubleshooting

**"Your branch is ahead of 'origin/main' by N commits"**
```bash
# Push your commits
git push origin main
```

**"Updates were rejected because the remote contains work"**
```bash
# Pull remote changes first
git pull --rebase origin main
# Resolve conflicts if any
git push origin main
```

**"Nothing to commit, working tree clean"**
```bash
# All changes already committed
# Push if needed: git push origin main
```

---

**Quick Reference Card** (Keep Visible):

```
Daily:  git status → git add [files] → git commit → git push
Phase:  Complete → Test → Attest → Update Docs → Tag → Push
Fix:    Test → Fix → Test → Commit → Push
Emergency: git reset --hard HEAD~1 (local only!)
```

**Current Status**: 12 commits ready to push to `tigantic/HyperTensor`

**Next Action**: Authenticate and push with `git push origin main`
