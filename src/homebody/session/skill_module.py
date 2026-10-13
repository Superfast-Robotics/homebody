"""Load a customized skill: one Python module in the skill protocol (NEEDS, FAILURES, PROMPT, prepare,
execute, optional abort), the same shape as skills/<name>.py, so a built-in skill's source is its
starting point. Its imports are limited to computation and the robot-side packages; `lib.<module>`
resolves to the task's own library folder. Relative imports resolve inside homebody.skills.
"""
import builtins
import hashlib
import importlib
import sys
import types
from pathlib import Path

from homebody.skills.contract import declared

PACKAGE = "homebody.skills"
ALLOWED = (
    "numpy", "scipy", "math", "cmath", "dataclasses", "functools", "itertools", "collections", "typing",
    "enum", "heapq", "bisect", "statistics", "copy", "operator", "abc", "re", "json",
    "homebody.skills", "homebody.motion", "homebody.helpers", "homebody.toolkit", "homebody.harness",
    "homebody.primitives", "homebody.vlm.topdown",
)
WITHHELD = ("open", "exec", "eval", "compile", "input", "breakpoint", "exit", "quit", "help", "__import__")
LIB = "lib"


class SkillLoadError(Exception):
    pass


def allowed(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in ALLOWED)


def digest(paths: list[Path]) -> str:
    """One hash over the files' names and contents, for 'this exact code was checked'."""
    hasher = hashlib.sha256()
    for path in sorted(paths):
        hasher.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return hasher.hexdigest()


class Loader:
    """Compiles skill modules against LIBRARY (the task's lib folder); every module it loads shares
    one restricted import function, so a lib module obeys the same allowlist."""

    def __init__(self, library: Path):
        self.library = library
        self.libs: dict[str, types.ModuleType] = {}

    def guarded_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        if level:
            package = (globals or {}).get("__package__") or PACKAGE
            if package != PACKAGE:
                raise ImportError(f"relative imports are only for homebody.skills, not {package}")
            base = importlib.import_module(PACKAGE)
            if not name:
                for item in fromlist or ():
                    importlib.import_module(f"{PACKAGE}.{item}")
                return base
            return importlib.import_module(f"{PACKAGE}.{name}")
        if name == LIB or name.startswith(LIB + "."):
            return self.lib_import(name, fromlist)
        if not allowed(name):
            raise ImportError(f"import of {name!r} is not allowed in a skill; allowed: {', '.join(ALLOWED)}, lib.*")
        return builtins.__import__(name, globals, locals, fromlist, level)

    def lib_import(self, name, fromlist):
        parts = name.split(".")
        if len(parts) == 1:
            package = types.ModuleType(LIB)
            for item in fromlist or ():
                setattr(package, item, self.lib_module(item))
            return package
        if len(parts) != 2:
            raise ImportError("lib modules are flat: lib.<module>")
        module = self.lib_module(parts[1])
        if fromlist:
            return module
        package = types.ModuleType(LIB)
        setattr(package, parts[1], module)
        return package

    def lib_module(self, stem):
        if stem not in self.libs:
            path = self.library / f"{stem}.py"
            if not path.is_file():
                raise ImportError(f"no lib/{stem}.py in this task's library")
            self.libs[stem] = self.compiled(f"homebody_task_lib.{stem}", path.read_text(), path, package=LIB)
        return self.libs[stem]

    def namespace_builtins(self):
        table = {key: value for key, value in vars(builtins).items() if key not in WITHHELD}
        table["__import__"] = self.guarded_import
        return table

    def compiled(self, module_name, source, path, package=PACKAGE):
        module = types.ModuleType(module_name)
        module.__dict__.update(__builtins__=self.namespace_builtins(), __file__=str(path), __package__=package)
        sys.modules[module_name] = module  # dataclasses look their module up here
        try:
            exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102 -- the sub-agent's skill, restricted imports
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        return module

    def load(self, name: str, path: Path, version: int) -> types.ModuleType:
        """The skill module at PATH declared under NAME, or SkillLoadError with the reason."""
        try:
            module = self.compiled(f"homebody_task_skill.{name}_v{version}", path.read_text(), path)
            return declared(name, module)
        except SyntaxError as error:
            raise SkillLoadError(f"SyntaxError: {error.msg} (line {error.lineno})") from error
        except Exception as error:  # any failure to define the skill is the coder's feedback
            raise SkillLoadError(f"{type(error).__name__}: {error}") from error
