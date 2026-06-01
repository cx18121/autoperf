"""Static validation of generated bot code via `ast`.

Correctness/scope guard only — NOT a security sandbox (see spec Trust model).
Ensures a candidate imports only the allowed modules, defines `choose_move`,
and has no module-level executable statements (which could run on import).
"""

import ast

ALLOWED_IMPORT_ROOTS = {"numpy", "optimizer"}  # `optimizer.bot_api`, `numpy[.*]`
# Module-level statements that are allowed (everything else is a side effect).
# Bare expressions (ast.Expr) are handled separately so a docstring can be
# allowed while other expression statements are rejected as side effects.
_ALLOWED_TOPLEVEL = (
    ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
    ast.ClassDef, ast.Assign, ast.AnnAssign,
)


def _import_root(name: str) -> str:
    return name.split(".", 1)[0]


def validate_code(code: str) -> tuple[bool, str]:
    """Return (ok: bool, detail: str). detail is "" when ok."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e}"

    for node in tree.body:
        # Imports: every imported module root must be on the allowlist.
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _import_root(alias.name)
                if root not in ALLOWED_IMPORT_ROOTS:
                    return False, f"import not allowed: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = _import_root(node.module or "")
            if root not in ALLOWED_IMPORT_ROOTS:
                return False, f"import not allowed: {node.module}"
        # Module-level bare expressions are side effects, EXCEPT a docstring.
        elif isinstance(node, ast.Expr):
            if not (isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)):
                return False, "top-level side effect (module-level expression)"
        elif not isinstance(node, _ALLOWED_TOPLEVEL):
            return False, f"top-level statement not allowed: {type(node).__name__}"

    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if "choose_move" not in names:
        return False, "no module-level choose_move(board) function defined"

    return True, ""
