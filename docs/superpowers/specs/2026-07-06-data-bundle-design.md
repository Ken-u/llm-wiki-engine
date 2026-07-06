# Data Bundle Runtime Design

Date: 2026-07-06

## Summary

Add a `data_bundle` capability to the standalone runtime so a distributed runtime can load a packaged knowledge dataset in one step. The target protection level is the strongest local distribution mode this project can reasonably support: encrypted bundle files, online license authorization, short-lived runtime keys, and no full plaintext knowledge directory on disk.

This design does not claim absolute DRM. If an attacker fully controls the host while the runtime is answering questions, they can eventually observe plaintext through memory inspection, function hooks, patched binaries, or repeated authorized queries. The goal is to prevent casual file copying, prevent direct plaintext directory leakage, make bundle reuse dependent on license authorization, and raise the cost of bulk export.

## Goals

- Support a single `data_bundle` setting that configures the main knowledge base and optional case library.
- Keep the current filesystem runtime mode working unchanged.
- Add an encrypted bundle backend that does not extract a complete plaintext `knowledge/` or `cases/` directory.
- Require online authorization before decrypting protected bundle content.
- Allow license revocation, device binding, bundle identity checks, and short-lived key issuance.
- Restrict runtime APIs that currently expose full wiki trees, full wiki pages, and raw source files.
- Preserve the existing runtime user experience for chat, search, case search, and OpenAI-compatible chat as much as possible.

## Non-Goals

- No guarantee against a fully privileged local attacker extracting runtime plaintext.
- No remote-only knowledge service in the first implementation.
- No rewrite of the full engine ingest pipeline.
- No encrypted authoring workflow in the first implementation.
- No support for direct LanceDB or SQLite reads from inside an encrypted zip file.

## Current Runtime Constraints

The runtime is disk-first today:

- `RuntimeProject.disk_path` points to a real directory.
- Wiki tree and page APIs use `Path`, `rglob`, and `read_text`.
- BM25 scans markdown files under `wiki/`.
- Vector search uses LanceDB under `.llm-wiki/lancedb/`.
- Case search uses `cases.jsonl`, SQLite FTS, and LanceDB under `.llm-wiki/case-index/`.
- Agent tools read wiki and raw files directly from project directories.

Because LanceDB, SQLite, and many code paths expect real files, "compressed archive direct read" would force a large filesystem abstraction and still leave index files readable after extraction. The encrypted bundle backend should instead provide an explicit storage and retrieval interface.

## Architecture

Introduce a runtime storage boundary:

```text
Runtime API / Agent Tools / Search
  -> KnowledgeStore interface
    -> FilesystemKnowledgeStore
    -> EncryptedBundleKnowledgeStore
```

`FilesystemKnowledgeStore` adapts the current behavior and keeps using local project directories.

`EncryptedBundleKnowledgeStore` reads an encrypted bundle, obtains a short-lived key from the authorization server, decrypts only required objects into memory, and serves search/page/case operations through prebuilt encrypted indexes and encrypted content objects.

The runtime config chooses the backend:

```yaml
data_bundle:
  enabled: true
  backend: encrypted_bundle
  path: ./data/company.llmwiki-bundle
  auth_url: https://license.example.com/v1/runtime/authorize
  license_key: ${LLMWIKI_LICENSE_KEY}
  device_binding: true
  token_cache_ttl_seconds: 3600
  expose_full_content: false
```

When `data_bundle.enabled` is false, existing `knowledge.path` and `case_library.path` behavior remains the default.

## Bundle Format

Use one `.llmwiki-bundle` file. The file is a container with a small public header and encrypted payload objects:

```text
bundle-header.json
manifest.json
objects/
  wiki/<object_id>.enc
  raw/<object_id>.enc
  cases/<object_id>.enc
indexes/
  wiki_keyword.enc
  wiki_vector.enc
  case_keyword.enc
  case_vector.enc
signature
```

`bundle-header.json` contains only data needed before authorization:

- format version
- bundle id
- bundle version
- build timestamp
- signing key id
- encryption algorithm
- manifest hash
- ciphertext root hash

`manifest.json` is signed and may be public or partially encrypted. It must not contain sensitive body text. It contains:

- knowledge name
- model name
- optional case library name
- object counts
- index versions
- embedding model and dimensions
- object ids, content types, sizes, and hashes
- feature flags such as `has_cases`

Each object uses AES-256-GCM with a unique nonce. The content encryption key is never stored in plaintext in the bundle. The authorization server returns a short-lived unwrap response that lets the runtime derive or unwrap the key in memory.

## Authorization Flow

Startup flow:

1. Runtime loads the bundle header and manifest metadata.
2. Runtime computes the bundle hash.
3. Runtime builds a device fingerprint.
4. Runtime sends `license_key`, `bundle_id`, `bundle_hash`, runtime version, platform, and device fingerprint to `auth_url`.
5. Authorization server verifies license state, device policy, bundle entitlement, and revocation status.
6. Server returns a short-lived signed authorization token and key material needed to decrypt this bundle.
7. Runtime keeps key material only in memory.
8. Runtime opens the encrypted store and reports unlocked status through `/api/status`.

Failure behavior:

- Invalid license: runtime starts in locked mode and returns 423-style application errors for protected APIs.
- Network failure with no valid cached token: locked mode.
- Network failure with valid cached token and allowed offline grace: unlock until token expiry.
- Bundle hash mismatch: fail closed.
- Signature mismatch: fail closed.

The initial design allows a short token cache, but cached token content must be encrypted with a machine-local secret where available. The data key itself must not be written to `runtime-config.yaml` or logs.

## KnowledgeStore Interface

The first interface should cover runtime read paths only:

- `status()`
- `list_wiki_tree()`
- `read_wiki_page(path, mode)`
- `search_wiki(query, top_k, mode)`
- `resolve_wiki_page(path)`
- `fast_lookup(term)`
- `search_cases(query, limit)`
- `read_case(case_id, section=None)`
- `read_case_source(case_id, mode)`
- `read_raw_source(ref_or_path, mode)`

`mode` controls exposure:

- `snippet`: return bounded snippets only.
- `summary`: return generated or prebuilt summaries.
- `full`: return full plaintext only if `expose_full_content` allows it.

Runtime routes and agent tools should use this interface instead of constructing `Path` objects directly.

## Search and Indexing

Encrypted bundle mode should not depend on LanceDB or SQLite files.

For first implementation:

- Wiki keyword search uses a prebuilt encrypted inverted index or compact BM25 document table.
- Wiki vector search uses precomputed vectors stored in an encrypted index file and loaded into memory after authorization.
- Case keyword search uses a prebuilt encrypted case keyword index.
- Case vector search uses precomputed encrypted vectors loaded into memory.

For small and medium bundles, an in-memory exact top-k vector scan is acceptable. If large bundles need approximate search later, add a bundle-native ANN index that can be decrypted into memory without exposing a directory of reusable index files.

Index records should map to object ids and byte ranges, not plaintext paths. Public display paths can be stable logical ids such as `wiki/entities/foo.md`, but the store remains the authority for resolving them.

## API Exposure Changes

When `encrypted_bundle` is active and `expose_full_content` is false:

- `/api/status` shows bundle id, version, authorization state, token expiry, and whether case library exists. It must not show local decrypted paths.
- `/api/wiki` returns either disabled, a shallow logical tree, or metadata only.
- `/api/wiki/{path}` returns summaries or snippets by default, not full markdown.
- `/api/raw/{path}` is disabled.
- `/api/skill/documents/content` requires a valid short-lived `source_ref` and returns only allowed content.
- `/api/cases/{case_id}` returns bounded case detail unless full content is explicitly enabled.
- `/api/search` and `/api/cases/search` return snippets and logical references.
- `/v1/chat/completions` remains supported and uses the encrypted store internally.

Add response metadata so the UI can distinguish `filesystem` and `encrypted_bundle` modes.

## Anti-Bulk-Export Controls

Encrypted bundle mode should include:

- Per-request maximum plaintext characters.
- Top-k limits enforced server-side.
- Source references with short expiry and signatures.
- Optional rate limiting per API key or local client identity.
- Audit events for search, page reads, raw reads, case reads, authorization failures, and high-volume access.
- Configurable policy for full page/raw access.

These controls are not a cryptographic boundary. They reduce accidental leakage and raise the cost of scripted extraction.

## Packaging and Build Workflow

Add a bundle packaging command in a later implementation plan. It should consume existing compiled project directories:

```text
knowledge/
  purpose.md
  wiki/
  raw/sources/
  .llm-wiki/lancedb/
cases/
  raw/sources/
  .llm-wiki/case-index/
```

The packer should:

1. Validate the project layout.
2. Build bundle-native keyword and vector indexes.
3. Assign object ids.
4. Encrypt content and index objects.
5. Write manifest and header.
6. Sign the bundle.
7. Produce a hash to register with the authorization server.

The runtime executable does not need to contain customer data. The bundle file is distributed separately.

## Migration Plan

1. Add config models for `data_bundle`.
2. Introduce `KnowledgeStore` and adapt current filesystem behavior behind it.
3. Move runtime routes to `KnowledgeStore`.
4. Move agent runtime tool reads to `KnowledgeStore`.
5. Implement authorization client and locked status.
6. Implement encrypted bundle reader.
7. Implement bundle-native keyword and vector search.
8. Implement case search and case read support.
9. Add API restrictions and audit events for encrypted mode.
10. Add packaging command and documentation.

This keeps the existing runtime functional during the transition and allows tests to prove parity between filesystem mode and bundle mode.

## Testing Strategy

Unit tests:

- Config parsing and environment expansion for `data_bundle`.
- Bundle header and manifest validation.
- Signature and hash failure cases.
- Authorization success, failure, expiry, and cached token behavior.
- Object encryption and decryption with nonce uniqueness.
- KnowledgeStore parity for wiki search, wiki read, fast lookup, case search, and case read.

API tests:

- Filesystem mode behavior remains unchanged.
- Encrypted bundle locked mode returns clear errors.
- Encrypted bundle unlocked mode supports chat/search/case search.
- Full wiki/raw APIs are restricted when `expose_full_content` is false.
- Source refs expire and cannot be guessed.

Integration tests:

- Package a small fixture bundle.
- Authorize through a mock license server.
- Run runtime against the bundle.
- Verify no plaintext `knowledge/` or `cases/` directory is created.

Security checks:

- No data key in logs.
- No data key in config responses.
- Path traversal attempts against logical paths fail.
- Tampered bundle bytes fail closed.
- Tampered manifest fails closed.

## Open Implementation Decisions

- Exact device fingerprint fields by platform.
- Whether to support a short offline grace period in the first version.
- Whether manifest should be fully public or split into public header plus encrypted private manifest.
- Initial rate limit defaults.
- Initial in-memory vector index format.
- Location and schema for local audit logs.

## Acceptance Criteria

- A runtime can be configured with `data_bundle.backend: encrypted_bundle`.
- Without valid authorization, protected runtime APIs do not expose bundle content.
- With valid authorization, chat, search, wiki snippets, and optional case search work from the bundle.
- Runtime does not extract a complete plaintext knowledge or case directory.
- Existing filesystem runtime tests continue to pass.
- Documentation clearly states the security boundary and limitations.
