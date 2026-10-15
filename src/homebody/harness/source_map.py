"""SOURCE_MAP.md for a coder call: an outline of the skill it edits (definitions with line numbers and
first docstring lines, and where each failure code is returned) and of the HomeBody source the skill
imports (file, line and signature of each imported name), so the coder can open the right lines
instead of paging through whole files."""
import ast
from pathlib import Path

MAX_DOC = 110  # characters of a docstring's first line
MAX_MODULE_NAMES = 30  # names listed for a module imported as a whole


def source_map(target: str, source: str, source_root: Path) -> str:
    """The map for skill TARGET whose current SOURCE imports from SOURCE_ROOT/homebody."""
    lines = [f"# Source map of `skills/{target}.py`", "",
             "Line numbers refer to the file as it is now; open just the lines you need "
             "(for example `sed -n 120,180p`).", ""]
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return "\n".join(lines + [f"`skills/{target}.py` could not be parsed: {error}"]) + "\n"
    lines += ["## Definitions", "", *outline(tree), ""]
    codes = failure_lines(tree)
    if codes:
        lines += ["## Failure codes and where they are returned", "",
                  *(f"- `{code}`: {', '.join(f'L{n}' for n in found) or 'not returned here'}"
                    for code, found in codes.items()), ""]
    imported = imports(tree, Path(source_root) / "homebody")
    if imported:
        lines += ["## Imported HomeBody source (read-only)", ""]
        for path, names in imported.items():
            lines += [f"### `{path.relative_to(Path(source_root) / 'homebody')}`", *module_entries(path, names), ""]
    return "\n".join(lines) + "\n"


def outline(tree: ast.Module) -> list[str]:
    entries = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            entries.append(f"- L{node.lineno} `{signature(node)}`{doc(node)}")
        elif isinstance(node, ast.ClassDef):
            entries.append(f"- L{node.lineno} class `{node.name}`{doc(node)}")
            entries += [f"  - L{item.lineno} `{node.name}.{signature(item)}`{doc(item)}"
                        for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return entries


def failure_lines(tree: ast.Module) -> dict[str, list[int]]:
    """FAILURES codes -> lines of string constants equal to them outside the FAILURES assignment."""
    declared, skip = [], set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FAILURES" for t in node.targets):
            try:
                declared = [code for code in ast.literal_eval(node.value) if isinstance(code, str)]
            except ValueError:
                return {}
            skip = set(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    found: dict[str, list[int]] = {code: [] for code in declared}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and node.value in found and node.lineno not in skip
                and node.lineno not in found[node.value]):
            found[node.value].append(node.lineno)
    return {code: sorted(lines) for code, lines in found.items()}


def imports(tree: ast.Module, package: Path) -> dict[Path, list[str] | None]:
    """Module files under PACKAGE that the skill imports -> the names taken from them (None: the whole module)."""
    found: dict[Path, list[str] | None] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level == 1:
            base = package / "skills"
        elif node.level == 0 and (node.module or "").startswith("homebody."):
            base = package
        else:
            continue
        parts = (node.module or "").split(".")[1 if node.level == 0 else 0:] if node.module else []
        directory = base.joinpath(*parts)
        if directory.with_suffix(".py").is_file():
            path = directory.with_suffix(".py")
            found.setdefault(path, [])
            entries = found[path]
            if entries is not None:
                entries += [alias.name for alias in node.names]
        else:
            for alias in node.names:
                module = directory / f"{alias.name}.py"
                if module.is_file():
                    found[module] = None
    return found


def module_entries(path: Path, names: list[str] | None) -> list[str]:
    tree = ast.parse(path.read_text())
    defined: dict[str, ast.stmt] = {node.name: node for node in tree.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.setdefault(target.id, node)
    if names is None:
        chosen = [name for name in defined if not name.startswith("_")][:MAX_MODULE_NAMES]
    else:
        chosen = [name for name in names if name in defined]
    return [entry(name, defined[name]) for name in chosen]


def entry(name: str, node: ast.stmt) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return f"- L{node.lineno} `{signature(node)}`{doc(node)}"
    if isinstance(node, ast.ClassDef):
        return f"- L{node.lineno} class `{name}`{doc(node)}"
    return f"- L{node.lineno} `{name}` (constant)"


def signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    return f"{node.name}({ast.unparse(node.args)})"


def doc(node: ast.AST) -> str:
    text = ast.get_docstring(node) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else None
    if not text:
        return ""
    first = text.strip().splitlines()[0]
    return f": {first[:MAX_DOC]}{'...' if len(first) > MAX_DOC else ''}"
