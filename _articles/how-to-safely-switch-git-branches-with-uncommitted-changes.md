---
title: "� How to Safely Switch Git Branches Without Losing Work"
categories: ["Git & Version Control"]
layout: default
---

# � How to Safely Switch Branches in Git Without Losing Work

In real-world development, it's common to be working on one task when an urgent issue or bug suddenly needs your attention. You’re in the middle of implementing a feature on one branch, but now you need to switch to another branch to fix something — quickly and safely.

So, how do you handle this **without losing your work** or creating a mess in your Git history?

This guide explains exactly what happens when you switch branches with uncommitted changes, and the recommended ways to manage your workflow.

---

## � Scenario

You're working on:

```bash
feature/login-ui
```

You’ve made several **uncommitted changes**, but the work is not complete. Suddenly, you're asked to fix a critical bug in another branch:

```bash
hotfix/form-validation
```

---

## ❓ What Happens If You Switch Branches With Uncommitted Changes?

If you run:

```bash
git checkout hotfix/form-validation
```

Git will check whether your current uncommitted changes can be safely brought over to the new branch.

### ✅ If there’s **no conflict**:

* Git will allow the checkout.
* Your uncommitted changes will remain in the working directory.
* You'll now be on `hotfix/form-validation` **with those same changes still applied**.

### ❌ If there’s a **conflict** (i.e., both branches modify the same files):

* Git will block the checkout and show an error like:

```
error: Your local changes to the following files would be overwritten by checkout:
```

---

## ✅ Recommended Solutions

### 1. � Use `git stash` (Best Practice)

The most reliable and clean method is to **stash your changes**, switch branches, and then **restore** them when you're ready.

#### Step-by-step:

```bash
# Save your in-progress work
git stash push -m "WIP on login-ui"

# Switch to the other branch
git checkout hotfix/form-validation

# Work on the bug, commit and push changes

# Go back to your original task
git checkout feature/login-ui

# Restore your stashed work
git stash pop
```

� You can stash multiple times and view them using:

```bash
git stash list
```

---

### 2. � Make a Temporary WIP Commit

If your changes are substantial and you want a record in history (or you're nervous about stashing), you can make a **temporary commit** and clean it up later.

#### Example:

```bash
git add .
git commit -m "WIP: Partial implementation of login-ui"

# Now switch to the bugfix branch
git checkout hotfix/form-validation

# After you’re done, come back
git checkout feature/login-ui

# Continue working, then squash or amend the WIP commit
```

> � Remember to squash or reword your WIP commit before merging!

---

### 3. �️ Discard the Changes (Only if Safe)

If you’re sure the changes aren’t needed, you can discard them:

```bash
git reset --hard
```

⚠️ **Dangerous!** This deletes all local changes permanently. Only use if you're absolutely certain.

---

## � Summary

| Situation                      | What to Do               |
| ------------------------------ | ------------------------ |
| Want to pause and resume later | `git stash push/pop`     |
| Want to save work in history   | Temporary WIP commit     |
| Changes are safe to discard    | `git reset --hard`       |
| Switching between tasks often  | Use a clean Git workflow |

---

## � Bonus: Use `git worktree` (Advanced)

If you regularly need to work on multiple branches simultaneously, consider using [Git worktrees](https://git-scm.com/docs/git-worktree) to **check out multiple branches in parallel** — each in its own folder.

```bash
git worktree add ../hotfix-form-validation hotfix/form-validation
```

---

## � Conclusion

Interruptions and task-switching are part of real-world development. Using `git stash` or WIP commits lets you handle them without fear of losing work or cluttering your history.

Stay clean, stay calm — and let Git work for you.
