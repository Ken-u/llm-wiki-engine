# Runtime Zip Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build zip-based runtime bundles that can be packaged with scripts and loaded directly by the runtime.

**Architecture:** Add `app.runtime.bundle` as the boundary for bundle packaging, safe extraction, cache reuse, and process-local metadata. `app.runtime_main` resolves `--bundle` before loading runtime config, and `app.runtime.status` reports bundle metadata. Thin shell and batch scripts call the same Python pack command.

**Tech Stack:** Python standard library `zipfile`, `hashlib`, `tempfile`, `shutil`, `argparse`, FastAPI runtime status tests, pytest.

---

### Task 1: Bundle Extraction Core

**Files:**
- Create: `app/runtime/bundle.py`
- Test: `tests/test_runtime_bundle.py`

- [ ] **Step 1: Write failing extraction tests**

Create tests covering bundled config, data-only bundle requiring external config, missing config without external config, cache reuse, path traversal rejection, POSIX absolute path rejection, Windows absolute path rejection, and `${RUNTIME_BUNDLE_DIR}` support through environment setup.

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_runtime_bundle.py -v`
Expected: FAIL because `app.runtime.bundle` does not exist.

- [ ] **Step 3: Implement minimal extraction code**

Create `BundleError`, `BundleInfo`, `prepare_runtime_bundle`, `get_runtime_bundle_info`, safe zip validation, SHA-256 hashing, cache directory selection, extraction marker reuse, and `RUNTIME_BUNDLE_DIR` environment setup.

- [ ] **Step 4: Run extraction tests**

Run: `pytest tests/test_runtime_bundle.py -v`
Expected: PASS.

### Task 2: Bundle Packaging Core and CLI

**Files:**
- Modify: `app/runtime/bundle.py`
- Test: `tests/test_runtime_bundle.py`

- [ ] **Step 1: Write failing packaging tests**

Add tests for packing knowledge, cases, config, and hooks into the expected layout, refusing overwrite without `--force`, allowing overwrite with `--force`, and excluding transient files.

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_runtime_bundle.py -v`
Expected: FAIL because pack functions are missing.

- [ ] **Step 3: Implement packaging code**

Add `pack_runtime_bundle`, sorted recursive file collection, transient-file exclusion, deterministic zip writing, CLI `pack` subcommand, and concise summary printing.

- [ ] **Step 4: Run packaging tests**

Run: `pytest tests/test_runtime_bundle.py -v`
Expected: PASS.

### Task 3: Runtime Startup and Status Integration

**Files:**
- Modify: `app/runtime_main.py`
- Modify: `app/runtime/status.py`
- Test: `tests/test_runtime_bundle.py`
- Test: `tests/test_runtime_api.py`

- [ ] **Step 1: Write failing integration tests**

Add tests that load config from an extracted bundle, load a data-only bundle with external config using `${RUNTIME_BUNDLE_DIR}`, and report bundle metadata in `/api/status`.

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_runtime_bundle.py tests/test_runtime_api.py -v`
Expected: FAIL because runtime startup/status do not know bundles.

- [ ] **Step 3: Implement integration code**

Add `--bundle` to `runtime_main`, resolve the bundle before `load_runtime_config`, set `RUNTIME_CONFIG` to the selected config path, and include bundle metadata in status output.

- [ ] **Step 4: Run integration tests**

Run: `pytest tests/test_runtime_bundle.py tests/test_runtime_api.py -v`
Expected: PASS.

### Task 4: Hook Error Clarity and Wrapper Scripts

**Files:**
- Modify: `app/runtime/hooks.py`
- Create: `scripts/build-runtime-bundle.sh`
- Create: `scripts/build-runtime-bundle.bat`
- Test: `tests/test_runtime_hooks.py`
- Test: `tests/test_runtime_bundle.py`

- [ ] **Step 1: Write failing tests**

Add a hook test for a missing configured script producing an error that names the hook and configured command. Add wrapper-script tests that verify both script files exist and invoke `app.runtime.bundle pack`.

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_runtime_hooks.py tests/test_runtime_bundle.py -v`
Expected: FAIL because error text and scripts are not implemented.

- [ ] **Step 3: Implement hook error and scripts**

Improve hook start failure messages and add shell/batch wrappers that forward all arguments to `python -m app.runtime.bundle pack`.

- [ ] **Step 4: Run targeted tests**

Run: `pytest tests/test_runtime_hooks.py tests/test_runtime_bundle.py -v`
Expected: PASS.

### Task 5: Documentation and Full Verification

**Files:**
- Modify: `runtime-config.example.yaml`
- Possibly modify: `README.md`

- [ ] **Step 1: Update documentation**

Document `--bundle`, `${RUNTIME_BUNDLE_DIR}`, and `scripts/build-runtime-bundle.*` usage.

- [ ] **Step 2: Run full targeted verification**

Run: `pytest tests/test_runtime_bundle.py tests/test_runtime_hooks.py tests/test_runtime_api.py -v`
Expected: PASS.

- [ ] **Step 3: Run broader regression suite if time permits**

Run: `pytest -v`
Expected: PASS or report unrelated failures with details.
