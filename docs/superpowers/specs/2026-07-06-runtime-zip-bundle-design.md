# Runtime Zip Bundle Design

Date: 2026-07-06

## Summary

Add a first-version runtime bundle feature for ordinary zip-based distribution. A bundle contains the runtime config, knowledge base, optional case library, and hook scripts. The runtime can start directly from a bundle, safely extract it to a local cache directory, load the bundled config, and reuse the existing filesystem-based runtime behavior.

This version is not encrypted and does not provide DRM. It is a convenience and deployment format for Windows, macOS, and Linux.

## Goals

- Add a `--bundle` startup option for `python -m app.runtime_main`.
- Support `.llmwiki-bundle` and `.zip` files using the standard zip format.
- Bundle optional `runtime-config.yaml`, `data/knowledge`, optional `data/cases`, and optional `hooks`.
- Resolve relative config paths from the extracted bundle directory.
- Reuse existing runtime search, wiki, case library, chat, OpenAI-compatible API, and hook behavior.
- Safely reject path traversal and absolute-path entries during extraction.
- Reuse an already extracted bundle when the bundle content hash is unchanged.
- Expose basic bundle metadata through runtime status.
- Add a cross-platform bundle packaging command and wrapper scripts.

## Non-Goals

- No encryption, license authorization, or anti-extraction protection in this version.
- No direct reading from zip files without extraction.
- No new bundle-native search or index format.
- No bundle authoring UI.
- No automatic rebuild of missing indexes during bundle load.
- No automatic ingest or compilation during packaging; inputs must already be compiled runtime directories.

## Bundle Format

The bundle is a zip archive with this layout:

```text
runtime-config.yaml
data/
  knowledge/
    wiki/
    raw/sources/
    .llm-wiki/lancedb/
  cases/
    .llm-wiki/case-index/
hooks/
  startup.sh
  startup.bat
```

Only `data/knowledge` is required. `runtime-config.yaml`, the case library, and hooks are optional.

If the bundle does not contain `runtime-config.yaml`, the user must provide an external config with `--config`. The runtime sets `RUNTIME_BUNDLE_DIR` to the extracted bundle directory before loading config, so the external config can point at bundled data with existing environment expansion:

```yaml
knowledge:
  path: ${RUNTIME_BUNDLE_DIR}/data/knowledge

case_library:
  enabled: true
  path: ${RUNTIME_BUNDLE_DIR}/data/cases
```

The recommended extension is `.llmwiki-bundle`, but `.zip` should work because the container is plain zip.

## Packaging Command

Add one Python packaging entrypoint and thin platform wrapper scripts:

```bash
python -m app.runtime.bundle pack \
  --knowledge ./data/knowledge \
  --cases ./data/cases \
  --config ./runtime-config.yaml \
  --hooks ./hooks \
  --output ./dist/customer.llmwiki-bundle
```

Wrapper scripts:

```bash
scripts/build-runtime-bundle.sh --knowledge ./data/knowledge --config ./runtime-config.yaml --output ./dist/customer.llmwiki-bundle
scripts/build-runtime-bundle.bat --knowledge .\data\knowledge --config .\runtime-config.yaml --output .\dist\customer.llmwiki-bundle
```

Packaging rules:

- `--knowledge` is required and must point at a compiled knowledge runtime directory.
- `--cases` is optional and must point at a compiled case library directory if provided.
- `--config` is optional. If provided, it is copied into the bundle as `runtime-config.yaml`.
- `--hooks` is optional. If provided, its contents are copied into the bundle under `hooks/`.
- Output parent directories are created automatically.
- Existing output files are overwritten only with `--force`.
- Archive entries use POSIX-style paths for cross-platform consistency.
- The packer writes deterministic zip entries where practical by sorting files.
- The packer should exclude transient files such as `.DS_Store`, `Thumbs.db`, `__pycache__`, `.pytest_cache`, and hidden temporary files.

The packer should validate that:

- The knowledge directory exists.
- `knowledge/wiki` exists.
- If `--cases` is provided, the cases directory exists.
- If `--config` is provided, it is a file.
- If `--hooks` is provided, it is a directory.
- The resulting archive contains `data/knowledge`.

The command prints a concise summary:

```text
Bundle written: dist/customer.llmwiki-bundle
Knowledge: data/knowledge
Cases: data/cases
Config: runtime-config.yaml
Hooks: hooks
SHA256: ...
```

## Startup Flow

Add a `--bundle` argument to `app.runtime_main`:

```bash
python -m app.runtime_main --bundle ./dist/customer.llmwiki-bundle
```

The startup flow is:

1. Validate that the bundle path exists and is a zip archive.
2. Compute a stable SHA-256 hash of the bundle file bytes.
3. Select an extraction directory under a runtime bundle cache, keyed by the hash.
4. If the hash directory already contains a complete extraction marker, reuse it.
5. Otherwise extract into a temporary sibling directory, validating every zip entry before writing.
6. Atomically replace or promote the temporary directory to the hash directory.
7. If `<extracted>/runtime-config.yaml` exists, load it.
8. If the bundled config is missing and the user provided `--config`, set `RUNTIME_BUNDLE_DIR` to the extracted directory and load that external config.
9. If the bundled config is missing and the user did not provide `--config`, fail startup with a clear message telling the user to either add `runtime-config.yaml` to the bundle or start with `--config`.
10. Set `RUNTIME_CONFIG` to the config path that will be loaded.
11. Continue with the existing runtime config, hook, and server startup flow.

The initial cache location should be deterministic and cross-platform:

- If `LLMWIKI_RUNTIME_BUNDLE_CACHE` is set, use it.
- Otherwise use a directory under the user's cache/home location.
- Fall back to a `.runtime-bundles` directory next to the bundle if a user cache directory is unavailable.

## Config Resolution

The existing runtime config loader already resolves relative paths from the config file's directory. Because the bundled config is loaded from the extracted directory, existing relative paths continue to work:

```yaml
knowledge:
  path: ./data/knowledge

case_library:
  enabled: true
  path: ./data/cases

hooks:
  enabled: true
  scripts:
    - name: startup
      command:
        linux: ["bash", "./hooks/startup.sh"]
        darwin: ["bash", "./hooks/startup.sh"]
        windows: ["cmd", "/c", ".\\hooks\\startup.bat"]
```

Hook commands run with the config directory as their current working directory, so bundled relative hook paths are supported without new hook behavior when the config is bundled.

If a config enables hooks and points at hook scripts that are not present, the existing hook process start failure should be converted into a clear user-facing message that includes the hook name, platform, configured command, and the fact that the script may be missing from the bundle or must be supplied by the external config environment.

## Extraction Safety

Extraction must reject unsafe entries before writing any file:

- Empty names.
- Absolute POSIX paths.
- Absolute Windows paths such as `C:\...`.
- UNC paths.
- Entries containing `..` path segments.
- Entries that resolve outside the extraction directory.

The extractor should skip directory entries after validation and create parent directories for files. It should avoid preserving platform-specific executable bits as a requirement; hook commands can invoke scripts through `bash`, `cmd`, or `python`.

## Runtime Status

When started from a bundle, `/api/status` should include:

```json
{
  "bundle": {
    "enabled": true,
    "path": "/path/to/customer.llmwiki-bundle",
    "hash": "sha256...",
    "extract_dir": "/cache/llm-wiki-runtime/bundles/sha256..."
  }
}
```

When not started from a bundle, `bundle.enabled` should be `false`.

## Components

- `app/runtime/bundle.py`
  - Validates bundle paths.
  - Computes hashes.
  - Chooses cache directories.
  - Performs safe extraction and reuse.
  - Packs runtime directories into a zip bundle.
  - Stores process-local bundle metadata for status reporting.

- `app/runtime_main.py`
  - Adds `--bundle`.
  - Resolves the bundle before `load_runtime_config`.
  - Keeps `--config` behavior unchanged when `--bundle` is not provided.

- `app/runtime/status.py`
  - Adds bundle metadata to status output.

- `runtime-config.example.yaml`
  - Documents the expected bundled relative path pattern.

- `scripts/build-runtime-bundle.sh`
  - Shell wrapper for `python -m app.runtime.bundle pack`.

- `scripts/build-runtime-bundle.bat`
  - Windows wrapper for `python -m app.runtime.bundle pack`.

## Error Handling

- Missing bundle path: fail startup with a clear error.
- Invalid zip: fail startup with a clear error.
- Missing `runtime-config.yaml` without `--config`: fail startup with a clear error telling the user to add the config to the bundle or pass `--config`.
- Missing `runtime-config.yaml` with `--config`: continue with the external config.
- Config-specified hook script missing from the bundle or external config environment: fail hook startup with a clear message naming the missing command/script. External config can reference bundled hooks with `${RUNTIME_BUNDLE_DIR}/hooks/...`.
- Unsafe zip entry: fail startup and leave no complete extraction marker.
- Extraction failure: remove the temporary extraction directory if possible.
- Existing valid extraction: reuse it without modifying files.
- Missing required pack input: fail with a clear command-line error.
- Existing output bundle without `--force`: fail with a clear command-line error.
- Packaging a config that references hooks not included in `--hooks`: warn rather than fail, because hooks may be supplied externally.

## Testing Strategy

Unit tests:

- Extracts a minimal bundle and returns the bundled config path.
- Reuses an extracted bundle with the same hash.
- Rejects `../evil` entries.
- Rejects absolute POSIX paths.
- Rejects absolute Windows paths.
- Rejects missing `runtime-config.yaml`.
- Allows missing `runtime-config.yaml` when `--config` supplies an external config.
- Expands `${RUNTIME_BUNDLE_DIR}` in external config paths after bundle extraction.
- Reports a clear startup failure when hooks are enabled but a configured bundled hook script is missing.
- Packages knowledge, optional cases, optional config, and optional hooks into the expected archive layout.
- Refuses to overwrite an existing output without `--force`.
- Wrapper scripts invoke the same Python packer on Unix-like systems and Windows.

API/runtime tests:

- Load runtime config through `prepare_runtime_bundle`.
- Verify knowledge and case paths resolve under the extracted directory.
- Verify `/api/status` reports bundle metadata.

Regression tests:

- Existing config-only runtime tests continue to pass.
- Existing hook tests continue to pass.
- Bundle packaging tests pass on the current platform.

## Acceptance Criteria

- `python -m app.runtime_main --bundle ./customer.llmwiki-bundle` starts from bundled config and data.
- `python -m app.runtime_main --bundle ./customer.llmwiki-bundle --config ./runtime-config.yaml` can start a data-only bundle with external config.
- External config can use `${RUNTIME_BUNDLE_DIR}` to reference bundled knowledge, cases, and hooks.
- A data-only bundle without `--config` fails with a message telling the user how to provide config.
- A config-specified missing hook script fails with a message that identifies the missing hook command.
- The same bundle works on Windows, macOS, and Linux with platform-specific hook commands.
- Relative knowledge, case, and hook paths resolve from the extracted bundle root.
- Unsafe zip entries are rejected.
- Existing `--config` startup behavior remains unchanged.
- Runtime status shows whether a bundle is active and where it was extracted.
- `scripts/build-runtime-bundle.sh` and `scripts/build-runtime-bundle.bat` can create a bundle with the expected layout.
