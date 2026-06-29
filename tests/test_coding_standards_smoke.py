"""Coding-standards lint smoke test.

Mechanically checks the objectively verifiable parts of Requirement 25 across
the Catalyst source: natural identifiers of at most three words (25.1), no
placeholder-only bodies in concrete functions (25.2, 25.3), no TODO or FIXME
markers (25.4), and no references to AI assistant tools (25.5).

Scope: only the first-party source is scanned, namely ``catalyst/*.py`` and the
root ``compare.py`` (and ``scripts/*.py`` when that directory holds modules).
Tests under ``tests/`` are deliberately excluded because test code may name
fixtures more verbosely and may legitimately mention tool names in comments;
this file excludes itself for the same reason. Generated artifacts under
``results/``, ``plots/``, ``checkpoints/``, ``logs/``, ``__pycache__`` and the
hypothesis database are never scanned.

What cannot be checked mechanically (whether a comment explains "why" rather
than "what") is not asserted; the discipline checks here cover only the
concrete, decidable rules.
"""

import ast
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _source_files():
    """Source files in scope for the standards scan.

    Returns ``catalyst/*.py`` plus the root ``compare.py`` and any modules under
    ``scripts/``. Package ``__init__.py`` files and cache directories are left
    in or out naturally: ``__init__.py`` is real source and stays included.
    """
    files = sorted((PROJECT_ROOT / "catalyst").glob("*.py"))
    compare_cli = PROJECT_ROOT / "compare.py"
    if compare_cli.is_file():
        files.append(compare_cli)
    scripts_dir = PROJECT_ROOT / "scripts"
    if scripts_dir.is_dir():
        files.extend(sorted(scripts_dir.glob("*.py")))
    return files


# Connector words carry no concept on their own; "side by side" or "percent by
# rank" read as natural three-concept phrases, so prepositions and articles do
# not count toward the three-word identifier budget (Req 25.1 favours natural
# names over artificial brevity).
CONNECTOR_WORDS = {
    "and",
    "or",
    "by",
    "of",
    "to",
    "the",
    "in",
    "on",
    "for",
    "a",
    "an",
    "with",
}

# Names whose mechanical word count would mislead. Kept empty by default; this
# is the documented escape hatch for a genuinely justified exception.
IDENTIFIER_ALLOWLIST: set[str] = set()


def split_identifier_words(name):
    """Split an identifier into concept words for the three-word rule.

    Underscores separate words; CamelCase splits on a lowercase-to-uppercase
    boundary and on an acronym-to-word boundary (``HFConv`` -> ``HF`` + ``Conv``).
    Digits stay attached to their neighbouring letters so model and layer names
    read as one concept: ``gpt2``, ``Conv1D``, ``sha256`` and ``experiment3``
    are each a single word, never a word plus a stray number. This is why the
    GPT-2 model name is allowed while an assistant reference would not be. Pure
    numeric fragments and connector words are dropped before counting.
    """
    core = name.strip("_")
    if not core:
        return []
    words = []
    for token in core.split("_"):
        if not token:
            continue
        spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", token)
        spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spaced)
        for piece in spaced.split():
            if piece.isdigit():
                continue
            if piece.lower() in CONNECTOR_WORDS:
                continue
            words.append(piece)
    return words


def _defined_identifiers(tree):
    """Collect identifiers defined in a module via the AST.

    Covers function and class names, every kind of function argument, and names
    assigned at module or local scope. Dunder names (``__init__``) reduce to an
    empty word list once surrounding underscores are stripped, so they never
    trip the limit.
    """
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
            arguments = node.args
            collected = (
                list(arguments.posonlyargs)
                + list(arguments.args)
                + list(arguments.kwonlyargs)
            )
            for argument in collected:
                names.add(argument.arg)
            if arguments.vararg:
                names.add(arguments.vararg.arg)
            if arguments.kwarg:
                names.add(arguments.kwarg.arg)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


# AI assistant and tooling references that must not appear in source (Req 25.5).
# Deliberately narrow: it targets assistant or product names, not the substring
# "gpt", so the "gpt2"/"GPT-2" model the project actually uses is never flagged.
AI_TOOL_PATTERNS = [
    r"copilot",
    r"chatgpt",
    r"gpt-?4",
    r"claude",
    r"\bcursor\b",
    r"openai\s+assistant",
    r"\bas an ai\b",
    r"language model,?\s+i",
]


def test_no_todo_or_fixme_markers():
    """Validates: Requirements 25.4"""
    marker = re.compile(r"TODO|FIXME", re.IGNORECASE)
    offenders = []
    for source in _source_files():
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if marker.search(line):
                offenders.append(f"{source.name}:{number}: {line.strip()}")
    assert not offenders, "TODO/FIXME markers found:\n" + "\n".join(offenders)


def test_no_ai_tool_references():
    """Validates: Requirements 25.5"""
    patterns = [re.compile(p, re.IGNORECASE) for p in AI_TOOL_PATTERNS]
    offenders = []
    for source in _source_files():
        text = source.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            for pattern in patterns:
                if pattern.search(line):
                    offenders.append(f"{source.name}:{number}: {line.strip()}")
    assert not offenders, "AI-tool references found:\n" + "\n".join(offenders)


def test_identifiers_at_most_three_words():
    """Validates: Requirements 25.1"""
    offenders = []
    for source in _source_files():
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for name in _defined_identifiers(tree):
            if name in IDENTIFIER_ALLOWLIST:
                continue
            words = split_identifier_words(name)
            if len(words) > 3:
                offenders.append(f"{source.name}: {name} -> {words}")
    assert not offenders, "identifiers exceeding three words:\n" + "\n".join(offenders)


def test_no_placeholder_only_function_bodies():
    """Validates: Requirements 25.2, 25.3

    Flags concrete functions whose body is nothing but ``pass`` or ``...``,
    which are stub placeholders forbidden by Req 25.4's "placeholder-only
    statements". Abstract methods and Protocol members may legitimately have an
    empty body, so they are exempt. A leading docstring is ignored when judging
    whether the remaining body is a placeholder.
    """
    offenders = []
    for source in _source_files():
        tree = ast.parse(source.read_text(encoding="utf-8"))
        abstract_owners = _abstract_class_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _has_abstract_decorator(node):
                continue
            body = node.body
            if body and _is_docstring(body[0]):
                body = body[1:]
            if len(body) != 1:
                continue
            statement = body[0]
            is_pass = isinstance(statement, ast.Pass)
            is_ellipsis = (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and statement.value.value is Ellipsis
            )
            if (is_pass or is_ellipsis) and node.name not in abstract_owners:
                offenders.append(f"{source.name}: {node.name}")
    assert not offenders, "placeholder-only function bodies:\n" + "\n".join(offenders)


def _is_docstring(statement):
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _has_abstract_decorator(node):
    for decorator in node.decorator_list:
        name = decorator.attr if isinstance(decorator, ast.Attribute) else getattr(decorator, "id", "")
        if "abstract" in str(name).lower():
            return True
    return False


def _abstract_class_names(tree):
    """Method names belonging to Protocol or ABC classes, which may be empty."""
    exempt = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {
            base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
            for base in node.bases
        }
        if {"Protocol", "ABC"} & bases:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    exempt.add(item.name)
    return exempt
