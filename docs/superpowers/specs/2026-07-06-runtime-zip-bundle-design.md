# Runtime Zip Bundle Design

Date: 2026-07-06

## Summary

Add a first-version runtime bundle feature for ordinary zip-based distribution. A bundle contains the runtime config, knowledge base, optional case library, and hook scripts. The runtime can start directly from a bundle, safely extract it to a local cache directory, load the bundled config, and reuse the existing filesystem-based runtime behavior.

This version is not encrypted and does not provide DRM. It is a convenience and deployment format for Windows, macOS, and Linux.

## Goals

- Add a `--bundle` startup option for `python -m app.runtime_main`.
- Support `.llmwiki-bundle` and `.zip` files using the standard zip format.
- Bundle `runtime-config.yaml`, `data/knowledge`, optional `data/cases`, and optional `hooks`.
- Resolve relative config paths from the extracted bundle directory.
- Reuse existing runtime search, wiki, case library, chat, OpenAI-compatible API, and hook behavior.
- Safely reject path traversal and absolute-path entries during extraction.
- Reuse an already extracted bundle when the bundle content hash is unchanged.
- Expose basic bundle metadata through runtime status.

## Non-Goals

- No encryption, license authorization, or anti-extraction protection in this version.
- No direct reading from zip files without extraction.
- No new bundle-native search or index format.
- No bundle authoring UI.
- No automatic rebuild of missing indexes during bundle load.

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

Only `runtime-config.yaml` and `data/knowledge` are required. The case library and hooks are optional.

The recommended extension is `.llmwiki-bundle`, but `.zip` should work because the container is plain zip.

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
7. Load `<extracted>/runtime-config.yaml`.
8. Set `RUNTIME_CONFIG` to the bundled config path.
9. Continue with the existing runtime config, hook, and server startup flow.

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

Hook commands run with the config directory as their current working directory, so bundled relative hook paths are supported without new hook behavior.

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
  - Stores process-local bundle metadata for status reporting.

- `app/runtime_main.py`
  - Adds `--bundle`.
  - Resolves the bundle before `load_runtime_config`.
  - Keeps `--config` behavior unchanged when `--bundle` is not provided.

- `app/runtime/status.py`
  - Adds bundle metadata to status output.

- `runtime-config.example.yaml`
  - Documents the expected bundled relative path pattern.

## Error Handling

- Missing bundle path: fail startup with a clear error.
- Invalid zip: fail startup with a clear error.
- Missing `runtime-config.yaml`: fail startup with a clear error.
- Unsafe zip entry: fail startup and leave no complete extraction marker.
- Extraction failure: remove the temporary extraction directory if possible.
- Existing valid extraction: reuse it without modifying files.

## Testing Strategy

Unit tests:

- Extracts a minimal bundle and returns the bundled config path.
- Reuses an extracted bundle with the same hash.
- Rejects `../evil` entries.
- Rejects absolute POSIX paths.
- Rejects absolute Windows paths.
- Rejects missing `runtime-config.yaml`.

API/runtime tests:

- Load runtime config through `prepare_runtime_bundle`.
- Verify knowledge and case paths resolve under the extracted directory.
- Verify `/api/status` reports bundle metadata.

Regression tests:

- Existing config-only runtime tests continue to pass.
- Existing hook tests continue to pass.

## Acceptance Criteria

- `python -m app.runtime_main --bundle ./customer.llmwiki-bundle` starts from bundled config and data.
- The same bundle works on Windows, macOS, and Linux with platform-specific hook commands.
- Relative knowledge, case, and hook paths resolve from the extracted bundle root.
- Unsafe zip entries are rejected.
- Existing `--config` startup behavior remains unchanged.
- Runtime status shows whether a bundle is active and where it was extracted.
