---
title: "Complete Guide to Using Xray Test Management in Jira"
categories: ["Software Testing & QA"]
layout: default
---

# Complete Guide to Using Xray Test Management in Jira

Many teams use **Jira for project management**, but Jira alone does not provide full **test case management**. To manage testing inside Jira, teams install a plugin called **Xray Test Management**.

Xray extends Jira with powerful features like **test cases, test executions, test plans, automation integration, and test coverage tracking**.

This guide explains **how to set up Xray and how QA teams typically work with it**.

---

## Step 1: Install Xray Test Management in Jira

Xray is installed from the **Atlassian Marketplace**.

### Steps

1. Go to **Jira Settings**
2. Click **Apps**
3. Click **Find new apps**
4. Search for:

```
Xray Test Management for Jira
```

5. Click **Install**
6. Wait until the installation finishes.

After installation, Jira will automatically add **testing features and issue types**.

---

## Step 2: Understand Xray Issue Types

Xray introduces several new issue types used for testing.

| Issue Type | Purpose |
|---|---|
| **Test** | Represents a test case |
| **Precondition** | Setup required before running tests |
| **Test Set** | Group of test cases |
| **Test Execution** | Running tests and storing results |
| **Test Plan** | Organizes executions for a release |

These issue types integrate directly with **Jira Stories, Tasks, and Bugs**.

---

## Step 3: Organize Tests Using Test Repository

The **Test Repository** is a folder structure used to organize test cases.

Example structure:

```plaintext
Test Repository
 └ UAT Test Repository
     └ Manual Regression
         └ HUB
            └ Staysure
               └ HUB Quote
```

### Features of Test Repository

* Create folders
* Move tests between folders
* Organize regression tests
* Maintain large test suites easily

---

## Step 4: Create a Test Case

To create a test case:

1. Click **Create**
2. Select issue type **Test**
3. Fill the fields

Example:

```
Summary: Validate HUB quote journey mandatory fields
Test Type: Manual
Priority: High
Component: Staysure
```

Then add **Test Steps**.

Example test step:

| Step | Action                   | Data | Expected Result                          |
| ---- | ------------------------ | ---- | ---------------------------------------- |
| 1    | Click checkbox at bottom | None | Mandatory fields show red error messages |
| 2    | Fill mandatory fields    | None | Form accepts values successfully         |

Each test can contain **multiple steps with expected results**.

---

## Step 5: Execute Tests

Tests are executed through **Test Execution issues**.

### Steps

1. Create issue → **Test Execution**
2. Add tests to the execution
3. Run the tests manually or automatically

Possible results:

```
PASS
FAIL
TODO
BLOCKED
```

Example:

| Test Case          | Result |
| ------------------ | ------ |
| Login validation   | PASS   |
| Payment validation | FAIL   |

Execution results are stored inside Jira.

---

## Step 6: Use Test Coverage

Test Coverage connects **requirements with tests and results**.

Example relationship:

```plaintext
User Story
   ↓
Linked Test Cases
   ↓
Test Executions
   ↓
Execution Results
```

Coverage helps teams answer questions like:

* Are all requirements tested?
* Which stories still lack tests?
* Which features failed testing?

Example coverage view:

| Requirement      | Tests   | Status          |
| ---------------- | ------- | --------------- |
| Login Feature    | 3 tests | 2 PASS, 1 FAIL  |
| Checkout Feature | 2 tests | NOT RUN         |

---

## Step 7: Link Tests to Jira Stories

Tests should be linked to **requirements or user stories**.

Steps:

1. Open a **Story / Task**
2. Find **Test Coverage section**
3. Click **Add Tests**
4. Select or create test cases

Now the story automatically shows:

* Linked tests
* Execution results
* Coverage status

---

## Step 8: Create Test Plans

A **Test Plan** organizes testing for a specific release.

Example:

```
Release 2.0 Regression Plan
```

Contains:

```
Test Execution 1
Test Execution 2
Test Execution 3
```

Benefits:

* Manage release testing
* Track progress
* View pass/fail statistics

---

## Step 9: Integrate Automation Tests

Xray supports automation frameworks such as:

* Selenium
* Playwright
* Cypress
* JUnit
* Cucumber

Typical CI/CD pipeline:

```plaintext
1. Run automated tests in CI pipeline
2. Generate test report (JUnit or Cucumber JSON)
3. Upload report to Xray using API
4. Xray updates test execution results
```

This allows automated test results to appear inside Jira.

---

## Typical QA Workflow Using Xray

Most teams follow this workflow:

```plaintext
1. Create Requirement (Jira Story)
2. Create Test Cases
3. Organize Tests in Repository
4. Link Tests to Stories
5. Create Test Execution
6. Run Tests
7. View Test Coverage
8. Track Results in Test Plans
```

---

## Best Practices for Organizing Tests

Recommended folder structure:

```plaintext
Test Repository
 ├ Smoke Tests
 ├ Regression Tests
 ├ API Tests
 ├ UI Tests
 ├ Performance Tests
```

And create **Test Plans per release**.

Example:

```
Release 1.0 Plan
Release 1.1 Plan
Hotfix Plan
```

This keeps large projects manageable.

---

## Final Thoughts

**Xray Test Management** turns Jira into a powerful **test management platform**.
It enables teams to manage manual testing, automation results, and requirement coverage in one place.

By organizing tests properly and linking them to requirements, teams gain:

* Better visibility
* Higher test coverage
* Faster release cycles
* Better collaboration between QA and developers
