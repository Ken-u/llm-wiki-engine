"""Analysis and Generation prompt builders, ported from ingest.ts."""

from __future__ import annotations

from datetime import date

WIKI_TYPES = ["entity", "concept", "source", "query", "comparison", "synthesis", "overview"]


def _language_rule(source_content: str) -> str:
    """Heuristic language directive based on source content."""
    if not source_content:
        return ""
    sample = source_content[:2000]
    cjk_count = sum(1 for ch in sample if "\u4e00" <= ch <= "\u9fff")
    ratio = cjk_count / max(len(sample), 1)
    if ratio > 0.1:
        return "**Mandatory output language: 中文 (Chinese).** ALL output — frontmatter values, headings, body text — MUST be in Chinese."
    return "**Mandatory output language: English.** ALL output MUST be in English."


def build_analysis_prompt(purpose: str, index: str, source_content: str = "") -> str:
    parts = [
        "You are an expert research analyst. Read the source document and produce a structured analysis.",
        "Do not output chain-of-thought, hidden reasoning, or a thinking transcript. Reason internally and write only the concise final analysis.",
        "",
        _language_rule(source_content),
        "",
        "Your analysis should cover:",
        "",
        "## Key Entities",
        "List people, organizations, products, datasets, tools mentioned. For each:",
        "- Name and type",
        "- Role in the source (central vs. peripheral)",
        "- Whether it likely already exists in the wiki (check the index)",
        "",
        "## Key Concepts",
        "List theories, methods, techniques, phenomena. For each:",
        "- Name and brief definition",
        "- Why it matters in this source",
        "- Whether it likely already exists in the wiki",
        "",
        "## Main Arguments & Findings",
        "- What are the core claims or results?",
        "- What evidence supports them?",
        "- How strong is the evidence?",
        "",
        "## Connections to Existing Wiki",
        "- What existing pages does this source relate to?",
        "- Does it strengthen, challenge, or extend existing knowledge?",
        "",
        "## Contradictions & Tensions",
        "- Does anything in this source conflict with existing wiki content?",
        "- Are there internal tensions or caveats?",
        "",
        "## Recommendations",
        "- What wiki pages should be created or updated?",
        "- What should be emphasized vs. de-emphasized?",
        "- Any open questions worth flagging for the user?",
        "",
        "Be thorough but concise. Focus on what's genuinely important.",
        "",
        "If a folder context is provided, use it as a hint for categorization.",
    ]
    if purpose:
        parts.append(f"\n## Wiki Purpose (for context)\n{purpose}")
    if index:
        parts.append(f"\n## Current Wiki Index (for checking existing content)\n{index}")
    return "\n".join(p for p in parts if p is not None)


def build_generation_prompt(
    schema: str,
    purpose: str,
    index: str,
    source_filename: str,
    overview: str = "",
    source_content: str = "",
    source_summary_path: str | None = None,
) -> str:
    source_base = source_filename.rsplit(".", 1)[0] if "." in source_filename else source_filename
    summary_path = source_summary_path or f"wiki/sources/{source_base}.md"
    today = date.today().isoformat()

    parts = [
        "You are a wiki maintainer. Based on the analysis provided, generate wiki files.",
        "Do not output chain-of-thought, hidden reasoning, or explanatory preamble. Reason internally and output only the requested FILE/REVIEW blocks.",
        "",
        _language_rule(source_content),
        "",
        f"## IMPORTANT: Source File",
        f"The original source file is: **{source_filename}**",
        f"All wiki pages generated from this source MUST include this filename in their frontmatter `sources` field.",
        "",
        "## What to generate",
        "",
        f"1. A source summary page at **{summary_path}** (MUST use this exact path)",
        "2. Entity pages in wiki/entities/ for key entities identified in the analysis",
        "3. Concept pages in wiki/concepts/ for key concepts identified in the analysis",
        "4. An updated wiki/index.md — add new entries to existing categories, preserve all existing entries",
        f"5. A log entry for wiki/log.md (just the new entry to append, format: ## [{today}] ingest | Title)",
        "6. An updated wiki/overview.md — a high-level summary of what the entire wiki covers, updated to reflect the newly ingested source.",
        "",
        "## Frontmatter Rules (CRITICAL — parser is strict)",
        "",
        "Every page begins with a YAML frontmatter block. Format rules:",
        "",
        "1. The VERY FIRST line of the file MUST be exactly `---` (three hyphens, nothing else).",
        "2. Each frontmatter line is a `key: value` pair on its own line.",
        "3. The frontmatter ends with another `---` line on its own.",
        "4. Arrays use the standard YAML inline form `[a, b, c]`.",
        "",
        "Required fields and types:",
        f"  - type     — one of: {' | '.join(WIKI_TYPES)}",
        "  - title    — string",
        f"  - created  — date in YYYY-MM-DD form (use {today})",
        f"  - updated  — same as created",
        "  - tags     — array of bare strings: `tags: [microbiology, ai]`",
        "  - related  — array of bare wiki page slugs: `related: [foo, bar-baz]`",
        f'  - sources  — array of source filenames; MUST include "{source_filename}".',
        "",
        "Use [[wikilink]] syntax in the BODY for cross-references between pages.",
        "Use kebab-case filenames.",
        "",
        "## Review block types",
        "",
        "After all FILE blocks, optionally emit REVIEW blocks for anything that needs human judgment:",
        "- contradiction / duplicate / missing-page / suggestion",
        "OPTIONS: Create Page | Skip",
        "",
        "## Output Format (MUST FOLLOW EXACTLY)",
        "",
        "Your ENTIRE response consists of FILE blocks followed by optional REVIEW blocks. Nothing else.",
        "",
        "FILE block template:",
        "```",
        "---FILE: wiki/path/to/page.md---",
        "(complete file content with YAML frontmatter)",
        "---END FILE---",
        "```",
        "",
        "REVIEW block template:",
        "```",
        "---REVIEW: type | Title---",
        "Description",
        "OPTIONS: Create Page | Skip",
        "PAGES: wiki/page1.md, wiki/page2.md",
        "---END REVIEW---",
        "```",
        "",
        "## Output Requirements (STRICT)",
        "",
        "1. The FIRST character of your response MUST be `-` (the opening of `---FILE:`).",
        "2. DO NOT output any preamble.",
        "3. DO NOT echo the analysis.",
        "4. Between blocks, use only blank lines.",
        "",
        "If you start with anything other than `---FILE:`, the entire response will be discarded.",
    ]
    if purpose:
        parts.append(f"\n## Wiki Purpose\n{purpose}")
    if schema:
        parts.append(f"\n## Wiki Schema\n{schema}")
    if index:
        parts.append(f"\n## Current Wiki Index (preserve all existing entries, add new ones)\n{index}")
    if overview:
        parts.append(f"\n## Current Overview (update to reflect new source)\n{overview}")

    parts.extend([
        "",
        "---",
        "",
        _language_rule(source_content),
    ])
    return "\n".join(p for p in parts if p is not None)


def build_merge_prompt() -> str:
    return "\n".join([
        "You are merging two versions of the same wiki page into one coherent document.",
        "Both versions describe the same entity / concept; one is already on disk,",
        "the other was just generated from a different source document.",
        "",
        "Output ONE merged version that:",
        "- Preserves every factual claim from both versions (do not drop content)",
        "- Eliminates redundancy when both versions state the same fact",
        "- Reorganizes sections so the structure is logical",
        "- Uses consistent markdown structure",
        "- Keeps `[[wikilink]]` references intact",
        "",
        "Output requirements:",
        "- The FIRST character of your response MUST be `-` (the opening of `---`)",
        "- Output the COMPLETE file: YAML frontmatter + body",
        "- No preamble, no analysis prose",
    ])
