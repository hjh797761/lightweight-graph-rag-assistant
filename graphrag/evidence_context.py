"""Assemble whole source blocks with a bounded, one-hop supporting context.

Scores belong exclusively to retrieved core chunks. Structural links describe
why text accompanies them; they do not establish the truth of that text.
"""

import math
from collections.abc import Callable, Iterable

from .models import EvidenceChunk, EvidenceLink, EvidenceOptions, EvidenceSelection


def _target_ids(link: EvidenceLink) -> tuple[str, ...]:
    return link.target_ids or ((link.target_id,) if link.target_id else ())


def _source_groups(chunks: list[EvidenceChunk]) -> list[list[EvidenceChunk]]:
    """Merge only verified overlapping intervals, retaining every underlying ID."""
    groups: list[list[EvidenceChunk]] = []
    for chunk in sorted(chunks, key=lambda c: (c.doc_id, c.source_start, c.source_end, c.id)):
        text = chunk.source_text or chunk.clean_text
        valid = chunk.source_end - chunk.source_start == len(text)
        if groups:
            group = groups[-1]
            start = group[0].source_start
            end = max(c.source_end for c in group)
            raw = _group_text(group)
            overlap = min(end, chunk.source_end) - chunk.source_start
            if (valid and group[0].doc_id == chunk.doc_id and overlap > 0
                    and len(raw) == end - start
                    and raw[chunk.source_start-start:chunk.source_start-start+overlap] == text[:overlap]):
                group.append(chunk)
                continue
        groups.append([chunk])
    order = {c.id: i for i, c in enumerate(chunks)}
    return sorted(groups, key=lambda g: min(order[c.id] for c in g))


def _group_text(group: list[EvidenceChunk]) -> str:
    text = group[0].source_text or group[0].clean_text
    end = group[0].source_end
    for chunk in group[1:]:
        raw = chunk.source_text or chunk.clean_text
        if chunk.source_end > end:
            text += raw[max(0, end-chunk.source_start):]
            end = chunk.source_end
    return text


def _render(chunks, roles, reasons, links, *, support_only=False) -> str:
    """Render final text or its supporting blocks with their actual separators.

    A mixed-role overlapping block is conservatively charged in full to the
    supplement allowance, since its source text is intentionally rendered once.
    """
    selected_ids = {c.id for c in chunks}
    role_by_id = dict(zip((c.id for c in chunks), roles))
    citation_by_id = {c.id: f"[S{i}]" for i, c in enumerate(chunks, 1)}
    blocks = []
    supporting_blocks = []
    previous_end = {}
    for group in _source_groups(chunks):
        headers = []
        for chunk in group:
            role = "核心证据" if role_by_id[chunk.id] == "core" else "补充语境"
            headers.append(f"{citation_by_id[chunk.id]} [{role} | {chunk.id} | 文档 {chunk.doc_id} | "
                           f"{' / '.join(chunk.title_path)} | {chunk.locator} | "
                           f"原文 {chunk.source_start}:{chunk.source_end}]")
            for reason in reasons.get(chunk.id, []):
                if reason["source_id"] in selected_ids:
                    label = "引用目标" if reason["kind"] == "explicit_reference" else "邻接语境"
                    headers.append(f"{label}（来源 {reason['source_id']}）：{reason['basis']}")
            for link in links:
                if link.source_id == chunk.id and link.kind == "explicit_reference":
                    targets = _target_ids(link)
                    covered = sum(target in selected_ids for target in targets)
                    if targets and covered < len(targets):
                        headers.append(f"引用目标部分覆盖：{covered}/{len(targets)}")
            if chunk.partial:
                headers.append("原文片段（非完整小节）")
        first = group[0]
        is_support = any(role_by_id[c.id] == "supplement" for c in group)
        # Separate noncontiguous blocks visibly; never silently join their text.
        if first.doc_id in previous_end and first.source_start > previous_end[first.doc_id]:
            gap = first.source_gap_before
            distance = first.source_start - previous_end[first.doc_id]
            gap_block = gap if len(gap) == distance else "[... omitted ...]"
            if is_support:
                supporting_blocks.append(("\n\n" if blocks else "") + gap_block)
            blocks.append(gap_block)
        block = "\n".join(headers) + "\n" + _group_text(group)
        if is_support:
            supporting_blocks.append(("\n\n" if blocks else "") + block)
        blocks.append(block)
        previous_end[first.doc_id] = max(c.source_end for c in group)
    return "".join(supporting_blocks) if support_only else "\n\n".join(blocks)


def _load_support(loaded, cores, decisions):
    core_by_id = {c.id: c for c in cores}
    core_order = {c.id: i for i, c in enumerate(cores)}
    links, reasons, candidates = [], {}, {}
    for link, targets in loaded:
        if link.source_id not in core_by_id or link.kind not in ("explicit_reference", "adjacent"):
            continue
        if link.status != "resolved":
            decisions.append({"source_id": link.source_id, "kind": link.kind, "status": link.status})
            continue
        links.append(link)
        allowed = set(_target_ids(link))
        for target in targets:
            if target.id not in allowed or target.doc_id != core_by_id[link.source_id].doc_id:
                decisions.append({"chunk_id": target.id, "status": "invalid_target_scope"})
                continue
            reason = {"source_id": link.source_id, "kind": link.kind,
                      "basis": link.basis, "target_ids": list(_target_ids(link)),
                      "source_start": link.source_start, "source_end": link.source_end}
            target_reasons = reasons.setdefault(target.id, [])
            if reason not in target_reasons:
                target_reasons.append(reason)
            priority = (0 if link.kind == "explicit_reference" else 1,
                        core_order[link.source_id], target.id)
            if target.id not in candidates or priority < candidates[target.id][0]:
                candidates[target.id] = (priority, target)
    for values in reasons.values():
        values.sort(key=lambda r: (0 if r["kind"] == "explicit_reference" else 1,
                                   core_order[r["source_id"]], r["basis"], r["source_start"], r["source_end"]))
    return links, reasons, [c for _, c in sorted(candidates.values(), key=lambda item: item[0])]


def assemble_context(
    ranked_core: Iterable[tuple[EvidenceChunk, float]],
    link_loader: Callable[[list[str]], Iterable[tuple[EvidenceLink, list[EvidenceChunk]]]],
    *, options: EvidenceOptions, count_tokens: Callable[[str], int] | None = None,
) -> EvidenceSelection:
    """Select initial cores, load links once, then fill with unexpanded cores.

    ``count_tokens`` must be the caller's generator-tokenizer counter when token
    budgeting is requested. No truncation, embedding tokenizer, or score proxy
    is used. Decisions retain skipped/partial/unresolved selection provenance.
    """
    if options.budget_unit == "tokens" and count_tokens is None:
        raise ValueError("token budgeting requires generator-tokenizer count_tokens")
    counter = len if options.budget_unit == "characters" else count_tokens

    def cost(text):
        value = counter(text)
        if type(value) is not int or value < 0:
            raise ValueError("count_tokens must return a nonnegative integer")
        return value

    unique, seen = [], set()
    decisions = []
    for chunk, score in ranked_core:
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("core scores must be finite numbers")
        if chunk.id in seen:
            decisions.append({"chunk_id": chunk.id, "status": "duplicate_id"})
            continue
        unique.append((chunk, score))
        seen.add(chunk.id)
    result = EvidenceSelection(budget_unit=options.budget_unit, decisions=decisions)
    if not unique:
        return result
    supplement_cap = min(options.max_supplements, options.max_chunks - 1)
    if options.supplement_fraction == 0:
        supplement_cap = 0
    core_quota = options.max_chunks - supplement_cap
    initial_budget = options.context_budget * (1-options.supplement_fraction) if supplement_cap else options.context_budget
    chunks, roles, scores = [], [], []
    reasons, links = {}, []

    def render(candidate_chunks, candidate_roles):
        return _render(candidate_chunks, candidate_roles, reasons, links)

    def fits(candidate_chunks, candidate_roles):
        if cost(render(candidate_chunks, candidate_roles)) > options.context_budget:
            return False
        support = _render(candidate_chunks, candidate_roles, reasons, links, support_only=True)
        return not support or cost(support) <= options.context_budget * options.supplement_fraction

    for chunk, score in unique:
        if len(chunks) >= core_quota:
            break
        if cost(render(chunks + [chunk], roles + ["core"])) <= initial_budget:
            chunks.append(chunk)
            roles.append("core")
            scores.append(score)
            decisions.append({"chunk_id": chunk.id, "status": "selected", "phase": "initial_core"})
        else:
            decisions.append({"chunk_id": chunk.id, "status": "budget_skipped", "phase": "initial_core"})
    if chunks:
        links, reasons, candidates = _load_support(link_loader([c.id for c in chunks]), chunks, decisions)
        # Newly discovered annotations are charged too. A very long link reason
        # may consume the initial reserve; remove whole trailing blocks if needed.
        while chunks and cost(render(chunks, roles)) > options.context_budget:
            removed = chunks.pop()
            roles.pop()
            scores.pop()
            decisions.append({"chunk_id": removed.id, "status": "budget_skipped", "phase": "link_annotations"})
        for candidate in candidates:
            selected_ids = {c.id for c in chunks}
            if candidate.id in selected_ids:
                continue
            if not any(r["source_id"] in selected_ids for r in reasons[candidate.id]):
                continue
            if roles.count("supplement") >= supplement_cap or len(chunks) >= options.max_chunks:
                decisions.append({"chunk_id": candidate.id, "status": "cap_skipped", "phase": "supplement"})
                continue
            trial_chunks, trial_roles = chunks + [candidate], roles + ["supplement"]
            if fits(trial_chunks, trial_roles):
                chunks, roles = trial_chunks, trial_roles
                scores.append(None)
                decisions.append({"chunk_id": candidate.id, "status": "selected", "phase": "supplement"})
            else:
                decisions.append({"chunk_id": candidate.id, "status": "budget_skipped", "phase": "supplement"})

    for chunk, score in unique:
        if chunk.id in {c.id for c in chunks}:
            continue
        if len(chunks) >= options.max_chunks:
            decisions.append({"chunk_id": chunk.id, "status": "cap_skipped", "phase": "unexpanded_core"})
            continue
        if fits(chunks + [chunk], roles + ["core"]):
            chunks.append(chunk)
            roles.append("core")
            scores.append(score)
            decisions.append({"chunk_id": chunk.id, "status": "selected", "phase": "unexpanded_core"})
        else:
            decisions.append({"chunk_id": chunk.id, "status": "budget_skipped", "phase": "unexpanded_core"})
    selected_ids = {c.id for c in chunks}
    for link in links:
        if link.kind == "explicit_reference" and link.source_id in selected_ids:
            missing = [target for target in _target_ids(link) if target not in selected_ids]
            if missing:
                decisions.append({"source_id": link.source_id, "status": "partial_coverage",
                                  "target_ids": list(_target_ids(link)), "missing_ids": missing})
    result.chunks, result.roles, result.scores = chunks, roles, scores
    result.reasons = {id: [r for r in values if r["source_id"] in selected_ids]
                      for id, values in reasons.items() if id in selected_ids}
    result.context = render(chunks, roles)
    result.budget_used = cost(result.context) if result.context else 0
    result.status = "ok" if chunks else "budget_exhausted"
    return result
