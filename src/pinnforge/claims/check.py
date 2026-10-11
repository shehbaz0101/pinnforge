"""Match published claims to committed JSON evidence.

Numbers are compared at the precision of the published token. A percent
token is the evidence value times the claim's ``scale`` (100 for a percent
of a relative error). Full-precision tokens must equal the evidence float.
Shorter tokens must equal that float rounded to the token's significant
digits.

A superiority phrase (``best``, ``beats``, ``outperforms``, ``most accurate``,
``wins``, ``edge``, and the listed inflections) in a scanned file must sit
inside a registered claim. A claim that the FNO inverse is the best or most
accurate method on ``hard_ood`` is checked against the strong-baseline
ordering. Lower mean relative error is the more accurate method. The sensor-lift
solver is lower than both FNO arms, so that claim fails.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_REGISTRY = Path("docs/claims.yaml")

# Longer phrases first so "most accurate" is not split into a shorter hit.
PHRASES: tuple[str, ...] = (
    "most accurate",
    "outperforming",
    "outperforms",
    "outperform",
    "beating",
    "beats",
    "beat",
    "best",
    "wins",
    "win",
    "edge",
)

NUMBER_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])[+-]?(?:\d+\.\d+(?:[eE][+-]?\d+)?%?|\d+[eE][+-]?\d+%?)"
)
LEAF_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])[+-]?(?:\d+\.\d+(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+|\d+)"
)
FNO_WORD = re.compile(r"(?i)\b(?:fno|fourier neural operator|hybrid|operator)\b")
HARD_OOD_WORD = re.compile(r"(?i)hard[-_ ]?ood")
BEST_WORD = re.compile(r"(?i)\b(?:best|most accurate)\b")
EDGE_WORD = re.compile(r"(?i)\bedge\b")
DENIAL_WORD = re.compile(
    r"(?i)\b(?:not|no|never|neither|without)\b|\bdoes not\b|\bdo not\b|\bis not\b|\bisn't\b"
)


@dataclass(frozen=True)
class Failure:
    """One claim-check failure."""

    claim_id: str
    location: str
    message: str

    def format(self) -> str:
        ident = self.claim_id or "-"
        place = self.location or "-"
        return f"{ident}: {place}: {self.message}"


def check(root: Path | None = None, registry_path: Path | None = None) -> list[Failure]:
    """Return every claim failure under ``root``. An empty list is success."""

    repo = Path.cwd() if root is None else Path(root)
    path = Path(registry_path) if registry_path is not None else repo / DEFAULT_REGISTRY
    if not path.is_file():
        return [Failure("-", str(path), "claims registry was not found")]
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [Failure("-", str(path), f"claims registry is not valid YAML: {exc}")]
    if not isinstance(document, dict):
        return [Failure("-", str(path), "claims registry must be a mapping")]
    try:
        return _check_document(repo, document)
    except _RegistryError as exc:
        return [Failure(exc.claim_id, exc.location, exc.message)]


class _RegistryError(Exception):
    def __init__(self, message: str, claim_id: str = "-", location: str = "-") -> None:
        super().__init__(message)
        self.message = message
        self.claim_id = claim_id
        self.location = location


def _check_document(repo: Path, document: Mapping[str, Any]) -> list[Failure]:
    failures: list[Failure] = []
    cache = _EvidenceCache(repo)
    claims = document.get("claims")
    if not isinstance(claims, list) or not claims:
        raise _RegistryError("claims registry needs a non-empty claims list")
    seen: set[str] = set()
    ranking = _load_ranking(document.get("stronger_baselines"), cache)
    compiled: list[_Claim] = []
    for raw in claims:
        if not isinstance(raw, dict):
            raise _RegistryError("each claim must be a mapping")
        claim = _Claim.from_mapping(raw)
        if claim.claim_id in seen:
            raise _RegistryError(f"duplicate claim id {claim.claim_id}", claim.claim_id)
        seen.add(claim.claim_id)
        compiled.append(claim)
        try:
            failures.extend(claim.evaluate(repo, cache, ranking))
        except _RegistryError as exc:
            failures.append(Failure(exc.claim_id, exc.location, exc.message))

    failures.extend(_unregistered_phrases(repo, document, compiled))
    failures.extend(_unregistered_numbers(repo, document, compiled))
    failures.extend(_leaf_documents(repo, document, cache))
    return failures


@dataclass(frozen=True)
class _Ranking:
    """hard_ood inverse mean relative errors. Lower is more accurate."""

    values: Mapping[str, float]
    fno_methods: frozenset[str]

    def order_text(self) -> str:
        ordered = sorted(self.values.items(), key=lambda item: item[1])
        parts = [f"{name} {_format_sig(value, 3)}" for name, value in ordered]
        return " < ".join(parts)

    def fno_is_best(self, method: str) -> bool:
        if method == "fno_inverse":
            arms = [self.values[name] for name in self.fno_methods]
            others = [value for name, value in self.values.items() if name not in self.fno_methods]
            return max(arms) < min(others)
        if method not in self.values:
            raise _RegistryError(f"unknown hard_ood method {method}")
        claimed = self.values[method]
        others = [value for name, value in self.values.items() if name != method]
        return claimed < min(others)


def _load_ranking(raw: object, cache: _EvidenceCache) -> _Ranking | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise _RegistryError("stronger_baselines must be a mapping")
    evidence = raw.get("evidence")
    methods = raw.get("methods")
    fno_methods = raw.get("fno_methods")
    if not isinstance(evidence, str) or not isinstance(methods, dict) or not methods:
        raise _RegistryError("stronger_baselines needs evidence and methods")
    if not isinstance(fno_methods, list) or not fno_methods:
        raise _RegistryError("stronger_baselines needs fno_methods")
    values: dict[str, float] = {}
    for name, path in methods.items():
        values[str(name)] = float(cache.lookup(evidence, path))
    fno = frozenset(str(item) for item in fno_methods)
    missing = sorted(fno - set(values))
    if missing:
        raise _RegistryError(f"fno_methods not in stronger_baselines.methods: {missing}")
    return _Ranking(values, fno)


@dataclass(frozen=True)
class _Span:
    file: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class _Claim:
    claim_id: str
    kind: str
    locations: tuple[tuple[str, re.Pattern[str]], ...]
    numbers: tuple[Mapping[str, Any], ...]
    comparisons: tuple[Mapping[str, Any], ...]
    assertion: Mapping[str, Any] | None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> _Claim:
        claim_id = raw.get("id")
        kind = raw.get("class")
        if not isinstance(claim_id, str) or not claim_id:
            raise _RegistryError("claim id must be a non-empty string")
        if kind not in {"numeric", "comparison", "superiority"}:
            raise _RegistryError("claim class must be numeric, comparison, or superiority", claim_id)
        locations = _locations(raw, claim_id)
        numbers = raw.get("numbers") or ()
        comparisons = raw.get("comparisons") or ()
        assertion = raw.get("assertion")
        if numbers and not isinstance(numbers, list):
            raise _RegistryError("numbers must be a list", claim_id)
        if comparisons and not isinstance(comparisons, list):
            raise _RegistryError("comparisons must be a list", claim_id)
        if assertion is not None and not isinstance(assertion, dict):
            raise _RegistryError("assertion must be a mapping", claim_id)
        if kind == "superiority" and not locations:
            raise _RegistryError("a superiority claim needs a location pattern", claim_id)
        return cls(
            claim_id,
            str(kind),
            locations,
            tuple(numbers),
            tuple(comparisons),
            assertion,
        )

    def spans(self, repo: Path) -> list[_Span]:
        found: list[_Span] = []
        for relative, pattern in self.locations:
            text = _read(repo, relative, self.claim_id)
            for match in pattern.finditer(text):
                found.append(_Span(relative, match.start(), match.end(), match.group(0)))
        return found

    def evaluate(
        self,
        repo: Path,
        cache: _EvidenceCache,
        ranking: _Ranking | None,
    ) -> list[Failure]:
        failures: list[Failure] = []
        spans = self.spans(repo)
        if self.locations and not spans:
            rendered = ", ".join(f"{relative}: /{pattern.pattern}/" for relative, pattern in self.locations)
            failures.append(
                Failure(self.claim_id, rendered, "registered location did not match the file")
            )
        for spec in self.numbers:
            failures.extend(self._number(repo, cache, spec))
        for spec in self.comparisons:
            failures.extend(self._comparison(cache, spec))
        if self.kind == "superiority":
            failures.extend(self._superiority(spans, ranking))
        return failures

    def _number(self, repo: Path, cache: _EvidenceCache, spec: Mapping[str, Any]) -> list[Failure]:
        token = spec.get("token")
        if not isinstance(token, str) or not token:
            raise _RegistryError("numeric claim needs a token", self.claim_id)
        files = spec.get("files")
        if not isinstance(files, list) or not files:
            raise _RegistryError(f"token {token} needs files", self.claim_id)
        evidence = _resolve_number(cache, spec, self.claim_id)
        scaled = evidence * float(spec.get("scale", 1))
        if not _token_matches(token, scaled):
            return [
                Failure(
                    self.claim_id,
                    ", ".join(str(item) for item in files),
                    f"token {token} does not match evidence {evidence!r} "
                    f"(scaled {scaled!r}) from {spec.get('evidence')} {spec.get('path')}",
                )
            ]
        missing = [str(item) for item in files if token not in _read(repo, str(item), self.claim_id)]
        if missing:
            return [
                Failure(
                    self.claim_id,
                    ", ".join(missing),
                    f"token {token} is not in the listed file",
                )
            ]
        return []

    def _comparison(self, cache: _EvidenceCache, spec: Mapping[str, Any]) -> list[Failure]:
        label = str(spec.get("label") or self.claim_id)
        op = spec.get("op")
        if op not in {"<", ">", "<=", ">=", "==", "each<", "each>"}:
            raise _RegistryError(f"unsupported comparison op {op}", self.claim_id, label)
        left = _resolve_side(cache, spec.get("left"), self.claim_id, label)
        right = _resolve_side(cache, spec.get("right"), self.claim_id, label)
        if not _compare(op, left, right):
            return [
                Failure(
                    self.claim_id,
                    label,
                    f"comparison {label} failed: {left!r} {op} {right!r}",
                )
            ]
        return []

    def _superiority(self, spans: Sequence[_Span], ranking: _Ranking | None) -> list[Failure]:
        assertion = self.assertion or {}
        kind = assertion.get("kind")
        if kind not in {"ranking", "not_a_ranking", "best", "most_accurate", "denies_best"}:
            raise _RegistryError(
                "superiority assertion.kind must be ranking, not_a_ranking, best, "
                "most_accurate, or denies_best",
                self.claim_id,
            )
        failures: list[Failure] = []
        text = " ".join(span.text for span in spans)
        location = ", ".join(dict.fromkeys(span.file for span in spans)) or self.claim_id
        forced = _text_claims_fno_is_best(text)
        if forced and kind == "not_a_ranking":
            failures.append(
                Failure(
                    self.claim_id,
                    location,
                    "text claims an FNO advantage on hard_ood but the claim is registered "
                    "as not_a_ranking",
                )
            )
        effective = kind
        if forced and kind in {"ranking", "not_a_ranking"}:
            effective = "best"
        if effective in {"best", "most_accurate", "denies_best"}:
            if ranking is None:
                raise _RegistryError("stronger_baselines is required for this claim", self.claim_id)
            scope = assertion.get("scope")
            if scope not in {None, "hard_ood_inverse_mean_rel"}:
                raise _RegistryError(f"unsupported assertion scope {scope}", self.claim_id)
            method = str(assertion.get("method") or "fno_inverse")
            is_best = ranking.fno_is_best(method)
            order = ranking.order_text()
            if effective in {"best", "most_accurate"} and not is_best:
                failures.append(
                    Failure(
                        self.claim_id,
                        location,
                        "claim that the FNO inverse is the best or most accurate method on "
                        f"hard_ood contradicts the strong baselines ({order})",
                    )
                )
            if effective == "denies_best" and is_best:
                failures.append(
                    Failure(
                        self.claim_id,
                        location,
                        "claim denies that the FNO inverse is the most accurate method on "
                        f"hard_ood, but the strong baselines say otherwise ({order})",
                    )
                )
        elif kind == "ranking" and not self.comparisons:
            failures.append(
                Failure(
                    self.claim_id,
                    location,
                    "a ranking superiority claim needs a comparison that the evidence satisfies",
                )
            )
        return failures


def _text_claims_fno_is_best(text: str) -> bool:
    """True when text gives the FNO an unqualified hard_ood advantage."""

    if not text or not FNO_WORD.search(text) or not HARD_OOD_WORD.search(text):
        return False
    if not BEST_WORD.search(text) and not EDGE_WORD.search(text):
        return False
    if DENIAL_WORD.search(text):
        return False
    return True


def _locations(raw: Mapping[str, Any], claim_id: str) -> tuple[tuple[str, re.Pattern[str]], ...]:
    locations = raw.get("locations")
    if locations is None:
        locations = []
    if not isinstance(locations, list):
        raise _RegistryError("locations must be a list", claim_id)
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for item in locations:
        if not isinstance(item, dict):
            raise _RegistryError("location must be a mapping", claim_id)
        relative = item.get("file")
        pattern = item.get("pattern")
        if not isinstance(relative, str) or not isinstance(pattern, str):
            raise _RegistryError("location needs file and pattern", claim_id)
        try:
            compiled.append((relative, re.compile(pattern)))
        except re.error as exc:
            raise _RegistryError(f"invalid location pattern: {exc}", claim_id) from exc
    return tuple(compiled)


def _unregistered_phrases(
    repo: Path,
    document: Mapping[str, Any],
    claims: Sequence[_Claim],
) -> list[Failure]:
    scan = document.get("scan")
    if not isinstance(scan, dict):
        raise _RegistryError("scan must be a mapping")
    files = _expand(repo, scan.get("superiority"))
    phrase_re = re.compile(
        "|".join(rf"\b{re.escape(item)}\b" for item in PHRASES),
        re.IGNORECASE,
    )
    covered = [claim for claim in claims if claim.kind == "superiority"]
    failures: list[Failure] = []
    for relative in files:
        text = _read(repo, relative, "-")
        for match in phrase_re.finditer(text):
            if any(_overlaps(span, relative, match.start(), match.end()) for claim in covered for span in claim.spans(repo)):
                continue
            sentence = _sentence(text, match.start(), match.end())
            line = text.count("\n", 0, match.start()) + 1
            location = f"{relative}:{line}"
            if _text_claims_fno_is_best(sentence):
                failures.append(
                    Failure(
                        "-",
                        location,
                        "unregistered superiority phrase "
                        f"{match.group(0)!r} claims an FNO advantage on hard_ood. "
                        "The strong baselines do not support calling the FNO inverse the "
                        "best or most accurate method on that slice",
                    )
                )
            else:
                failures.append(
                    Failure(
                        "-",
                        location,
                        f"unregistered superiority phrase {match.group(0)!r}",
                    )
                )
    return failures


def _unregistered_numbers(
    repo: Path,
    document: Mapping[str, Any],
    claims: Sequence[_Claim],
) -> list[Failure]:
    scan = document.get("scan")
    if not isinstance(scan, dict):
        return []
    listed = scan.get("numbers")
    if listed is None:
        return []
    files = _expand(repo, listed)
    ignores = scan.get("number_ignore") or []
    if not isinstance(ignores, list):
        raise _RegistryError("scan.number_ignore must be a list")
    ignore_res = [re.compile(str(item)) for item in ignores]
    numeric = [claim for claim in claims if claim.numbers]
    failures: list[Failure] = []
    for relative in files:
        text = _read(repo, relative, "-")
        text = re.sub(
            r"<style\b[^>]*>.*?</style>",
            lambda match: " " * (match.end() - match.start()),
            text,
            flags=re.DOTALL,
        )
        ignored = [match.span() for pattern in ignore_res for match in pattern.finditer(text)]
        for match in NUMBER_TOKEN.finditer(text):
            if any(start <= match.start() and match.end() <= end for start, end in ignored):
                continue
            token = match.group(0)
            if _number_covered(numeric, relative, token, match.start(), match.end(), repo):
                continue
            line = text.count("\n", 0, match.start()) + 1
            failures.append(
                Failure("-", f"{relative}:{line}", f"unregistered number {token}")
            )
    return failures


def _number_covered(
    claims: Sequence[_Claim],
    relative: str,
    token: str,
    start: int,
    end: int,
    repo: Path,
) -> bool:
    for claim in claims:
        for spec in claim.numbers:
            if spec.get("token") != token:
                continue
            files = spec.get("files") or []
            if relative not in files:
                continue
            pattern = re.compile(re.escape(token))
            text = _read(repo, relative, claim.claim_id)
            if any(item.start() <= start and end <= item.end() for item in pattern.finditer(text)):
                return True
            # The token is listed for this file and occurs in it. That is enough
            # when the same token is the whole published number.
            if token in text:
                return True
    return False


def _leaf_documents(
    repo: Path,
    document: Mapping[str, Any],
    cache: _EvidenceCache,
) -> list[Failure]:
    listed = document.get("leaf_documents") or []
    if not isinstance(listed, list):
        raise _RegistryError("leaf_documents must be a list")
    failures: list[Failure] = []
    for item in listed:
        if not isinstance(item, dict):
            raise _RegistryError("leaf document must be a mapping")
        relative = item.get("file")
        evidence = item.get("evidence")
        if not isinstance(relative, str) or not isinstance(evidence, list) or not evidence:
            raise _RegistryError("leaf document needs file and evidence")
        formatted: set[str] = set()
        for source in evidence:
            formatted.update(cache.leaf_strings(str(source)))
        text = _read(repo, relative, "-")
        for match in LEAF_TOKEN.finditer(text):
            if text[max(0, match.start() - 4) : match.start()] == "SHA-":
                continue
            token = match.group(0)
            try:
                value = float(token)
            except ValueError:
                failures.append(Failure("-", relative, f"unparsed number {token}"))
                continue
            rendered = format(value, ".16g")
            integer = str(int(value)) if value.is_integer() else ""
            if rendered in formatted or (integer and integer in formatted) or token in formatted:
                continue
            line = text.count("\n", 0, match.start()) + 1
            failures.append(
                Failure(
                    "-",
                    f"{relative}:{line}",
                    f"number {token} is not a .16g leaf of {evidence}",
                )
            )
    return failures


def _token_matches(token: str, scaled: float) -> bool:
    shown = float(token[:-1] if token.endswith("%") else token)
    if math.isclose(shown, scaled, rel_tol=0.0, abs_tol=0.0):
        return True
    digits = _significant_digits(token)
    return _format_sig(shown, digits) == _format_sig(scaled, digits)


def _significant_digits(token: str) -> int:
    raw = token[:-1] if token.endswith("%") else token
    raw = raw.lower().lstrip("+-")
    mantissa = raw.split("e", 1)[0]
    digits = mantissa.replace(".", "").lstrip("0")
    return max(1, len(digits))


def _format_sig(value: float, digits: int) -> str:
    if value == 0.0:
        return f"{0.0:.{digits}g}"
    return f"{value:.{digits}g}"


def _compare(op: str, left: Any, right: Any) -> bool:
    if op in {"each<", "each>"}:
        if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
            return False
        pairs = zip((float(item) for item in left), (float(item) for item in right), strict=True)
        if op == "each<":
            return all(item < other for item, other in pairs)
        return all(item > other for item, other in pairs)
    left_value = float(left)
    right_value = float(right)
    if op == "<":
        return left_value < right_value
    if op == ">":
        return left_value > right_value
    if op == "<=":
        return left_value <= right_value
    if op == ">=":
        return left_value >= right_value
    return math.isclose(left_value, right_value, rel_tol=0.0, abs_tol=0.0)


def _resolve_side(cache: _EvidenceCache, raw: object, claim_id: str, label: str) -> Any:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if not isinstance(raw, dict):
        raise _RegistryError("comparison side must be a mapping or a number", claim_id, label)
    if "literal" in raw:
        return raw["literal"]
    value = _resolve_number(cache, raw, claim_id)
    minus = raw.get("minus")
    if minus is not None:
        value = float(value) - float(_resolve_number(cache, minus, claim_id))
    return value


def _resolve_number(cache: _EvidenceCache, spec: Mapping[str, Any], claim_id: str) -> Any:
    evidence = spec.get("evidence")
    if not isinstance(evidence, str):
        raise _RegistryError("evidence path is required", claim_id)
    path = spec.get("path")
    regex = spec.get("regex")
    if path is None and not isinstance(regex, str):
        raise _RegistryError("evidence needs a path or a regex", claim_id)
    if path is None:
        text = cache.text(evidence)
        return _extract_regex(text, str(regex), claim_id)
    value = cache.lookup(evidence, path)
    if isinstance(regex, str):
        if not isinstance(value, str):
            raise _RegistryError("regex extraction needs a string JSON value", claim_id)
        return _extract_regex(value, regex, claim_id)
    return value


def _extract_regex(text: str, pattern: str, claim_id: str) -> float:
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise _RegistryError(f"invalid evidence regex: {exc}", claim_id) from exc
    match = compiled.search(text)
    if match is None:
        raise _RegistryError(f"evidence regex did not match {pattern}", claim_id)
    group = match.group(1) if match.groups() else match.group(0)
    try:
        return float(group)
    except ValueError as exc:
        raise _RegistryError(f"evidence regex group is not a number: {group}", claim_id) from exc


def _overlaps(span: _Span, relative: str, start: int, end: int) -> bool:
    return span.file == relative and span.start < end and start < span.end


def _sentence(text: str, start: int, end: int) -> str:
    left = max(text.rfind(".", 0, start), text.rfind("\n", 0, start))
    right_dot = text.find(".", end)
    right_nl = text.find("\n", end)
    candidates = [item for item in (right_dot, right_nl) if item != -1]
    right = min(candidates) if candidates else len(text)
    return text[left + 1 : right + (1 if right_dot == right else 0)]


def _expand(repo: Path, listed: object) -> list[str]:
    if not isinstance(listed, list) or not listed:
        raise _RegistryError("scan list must be a non-empty list of paths")
    files: list[str] = []
    for item in listed:
        pattern = str(item)
        if any(char in pattern for char in "*?["):
            matches = sorted(path.relative_to(repo).as_posix() for path in repo.glob(pattern) if path.is_file())
            if not matches:
                raise _RegistryError(f"scan pattern matched no files: {pattern}")
            files.extend(matches)
        else:
            files.append(pattern)
    return files


def _read(repo: Path, relative: str, claim_id: str) -> str:
    path = repo / relative
    if not path.is_file():
        raise _RegistryError(f"file not found: {relative}", claim_id, relative)
    return path.read_text(encoding="utf-8")


class _EvidenceCache:
    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self._json: dict[str, Any] = {}
        self._text: dict[str, str] = {}
        self._leaves: dict[str, set[str]] = {}

    def text(self, relative: str) -> str:
        if relative not in self._text:
            path = self._resolve(relative)
            self._text[relative] = path.read_text(encoding="utf-8")
        return self._text[relative]

    def lookup(self, relative: str, path: object) -> Any:
        if relative.endswith(".json"):
            node: Any = self._load_json(relative)
            for part in _path_parts(path):
                if isinstance(node, dict) and part in node:
                    node = node[part]
                elif isinstance(node, list) and part.isdigit():
                    node = node[int(part)]
                else:
                    raise _RegistryError(f"JSON path {path!r} missed at {part!r} in {relative}")
            return node
        return _extract_regex(self.text(relative), _path_as_regex(path), "-")

    def leaf_strings(self, relative: str) -> set[str]:
        if relative not in self._leaves:
            leaves: set[str] = set()
            _collect_leaves(self._load_json(relative), leaves)
            self._leaves[relative] = leaves
        return self._leaves[relative]

    def _load_json(self, relative: str) -> Any:
        if relative not in self._json:
            self._json[relative] = json.loads(self.text(relative))
        return self._json[relative]

    def _resolve(self, relative: str) -> Path:
        path = Path(relative)
        if not path.is_absolute():
            path = self.repo / relative
        if not path.is_file():
            raise _RegistryError(f"evidence file not found: {relative}", location=relative)
        return path


def _path_parts(path: object) -> list[str]:
    if isinstance(path, list):
        return [str(item) for item in path]
    if isinstance(path, str):
        return [part for part in path.split("/") if part]
    raise _RegistryError(f"JSON path must be a string or a list, not {path!r}")


def _path_as_regex(path: object) -> str:
    if isinstance(path, str):
        return path
    raise _RegistryError("non-JSON evidence needs regex")


def _collect_leaves(node: Any, leaves: set[str]) -> None:
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        if isinstance(node, float) and not math.isfinite(node):
            return
        value = float(node)
        leaves.add(format(value, ".16g"))
        if value.is_integer():
            leaves.add(str(int(value)))
        return
    if isinstance(node, dict):
        for item in node.values():
            _collect_leaves(item, leaves)
    elif isinstance(node, list):
        for item in node:
            _collect_leaves(item, leaves)
