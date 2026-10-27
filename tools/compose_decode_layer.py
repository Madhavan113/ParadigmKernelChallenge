#!/usr/bin/env python3
"""Compose muse_decode_layer/submission.py from the three standalone kernel files and the layer glue.

    python3 tools/compose_decode_layer.py [--attn FILE] [--moe FILE] [--proj FILE] [--glue FILE] [-o OUT] [--check]

Defaults: attention = muse_attention_decode/submission.py, MoE = muse_moe_w4a16/submission.py,
proj = muse_decode_layer/proj_fork.py (the layer's own fork of the bf16_proj module: it carries the sk2 PDL and the
fused pooler+router launch that the standalone file does not have; pass --proj muse_bf16_proj/submission.py only
once those were ported), glue = muse_decode_layer/layer_glue.py, output = muse_decode_layer/submission.py.

Each component is embedded whole (its CUDA sources, host code, prepare and kernel_fn), with EVERY module-level name
prefixed (att_, moe_, bfp_: `_launch` -> `att__launch`, `PDL` -> `moe_PDL`) by an AST walk that resolves scopes, so
locals, parameters, attributes, keyword arguments and string contents are untouched. The glue refers to the
prefixed names (att__launch, moe_kernel_fn, bfp_prepare, ...). The component's first line (`# okc: problem=...`) is
dropped; its attribution header and docstring stay.

--check: compile the result, then list names referenced at module scope (or as implicit globals in functions) that
no module-level binding, import or builtin provides: the glue referring to a component name that no longer exists.

    python3 tools/compose_decode_layer.py unprefix bfp_ FILE START END   # helper: strip a prefix from a line range
"""
import argparse
import ast
import builtins
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
          ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
COMPS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


# ----------------------------------------------------------------------------------------------- scope analysis
class Scope:
    def __init__(self, node, parent):
        self.node, self.parent = node, parent
        self.bound = set()       # names bound in this scope (params, stores, defs, imports)
        self.globals_ = set()    # names declared `global` here (functions only)


def _bind_target(t, scope):
    if isinstance(t, ast.Name):
        scope.bound.add(t.id)
    elif isinstance(t, (ast.Tuple, ast.List)):
        for e in t.elts:
            _bind_target(e, scope)
    elif isinstance(t, ast.Starred):
        _bind_target(t.value, scope)
    # Attribute / Subscript targets bind nothing


def _enclosing_non_comp(scope):
    while isinstance(scope.node, COMPS):
        scope = scope.parent
    return scope


class Binder(ast.NodeVisitor):
    """First pass: which names each scope binds. Scope objects are stored on the AST nodes (node._scope)."""

    def __init__(self, tree):
        self.scope = Scope(tree, None)
        tree._scope = self.scope
        self.module = self.scope

    def _enter(self, node):
        s = Scope(node, self.scope)
        node._scope = s
        self.scope = s
        return s

    def _leave(self):
        self.scope = self.scope.parent

    def visit_FunctionDef(self, node):
        self.scope.bound.add(node.name)
        for d in node.decorator_list:
            self.visit(d)
        a = node.args
        for d in a.defaults + [k for k in a.kw_defaults if k is not None]:
            self.visit(d)
        for arg in a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]:
            if arg.annotation:
                self.visit(arg.annotation)
        if node.returns:
            self.visit(node.returns)
        s = self._enter(node)
        for arg in a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]:
            s.bound.add(arg.arg)
        for st in node.body:
            self.visit(st)
        self._leave()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        a = node.args
        for d in a.defaults + [k for k in a.kw_defaults if k is not None]:
            self.visit(d)
        s = self._enter(node)
        for arg in a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]:
            s.bound.add(arg.arg)
        self.visit(node.body)
        self._leave()

    def visit_ClassDef(self, node):
        self.scope.bound.add(node.name)
        for d in node.decorator_list + node.bases + [k.value for k in node.keywords]:
            self.visit(d)
        self._enter(node)
        for st in node.body:
            self.visit(st)
        self._leave()

    def _visit_comp(self, node, elts):
        # the first iterable is evaluated in the enclosing scope
        self.visit(node.generators[0].iter)
        self._enter(node)
        for i, g in enumerate(node.generators):
            if i:
                self.visit(g.iter)
            _bind_target(g.target, self.scope)
            self.visit(g.target)
            for c in g.ifs:
                self.visit(c)
        for e in elts:
            self.visit(e)
        self._leave()

    def visit_ListComp(self, node):
        self._visit_comp(node, [node.elt])

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node):
        self._visit_comp(node, [node.key, node.value])

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.scope.bound.add(node.id)

    def visit_NamedExpr(self, node):
        self.visit(node.value)
        _bind_target(node.target, _enclosing_non_comp(self.scope))

    def visit_Global(self, node):
        self.scope.globals_.update(node.names)
        self.module.bound.update(node.names)

    def visit_Nonlocal(self, node):
        pass

    def visit_Import(self, node):
        for a in node.names:
            self.scope.bound.add((a.asname or a.name).split(".")[0])

    visit_ImportFrom = visit_Import

    def visit_ExceptHandler(self, node):
        if node.name:
            self.scope.bound.add(node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node):
        if node.name:
            self.scope.bound.add(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node):
        if node.name:
            self.scope.bound.add(node.name)


def resolves_to_module(name, scope):
    """True when `name`, referenced in `scope`, is the module-level binding (LEGB without builtins)."""
    inner = True
    s = scope
    while s.parent is not None:
        if isinstance(s.node, ast.ClassDef) and not inner:
            s = s.parent
            inner = False
            continue
        if name in s.globals_:
            return True
        if name in s.bound:
            return False
        s = s.parent
        inner = False
    return True


class Renamer(ast.NodeVisitor):
    """Second pass: every Name that resolves to a module-level binding in `names` -> an edit (line, col, end, new)."""

    def __init__(self, names, prefix, src_lines):
        self.names, self.prefix, self.lines = names, prefix, src_lines
        self.scope = None
        self.edits = []

    def _edit(self, lineno, col, end_col, old):
        self.edits.append((lineno, col, end_col, old, self.prefix + old))

    def _def_name(self, node, kw):
        # `def name` / `class name` at node position (decorators precede node.lineno)
        line = self.lines[node.lineno - 1]
        m = re.compile(rf"\b{kw}\s+({re.escape(node.name)})\b").search(line, node.col_offset)
        assert m, (kw, node.name, node.lineno)
        self._edit(node.lineno, m.start(1), m.end(1), node.name)

    def visit_Module(self, node):
        self.scope = node._scope
        for st in node.body:
            self.visit(st)

    def _visit_scoped(self, node, pre, body):
        for n in pre:
            self.visit(n)
        outer, self.scope = self.scope, node._scope
        for n in body:
            self.visit(n)
        self.scope = outer

    def visit_FunctionDef(self, node):
        if self.scope.parent is None and node.name in self.names:
            self._def_name(node, "def" if isinstance(node, ast.FunctionDef) else "async\\s+def")
        a = node.args
        pre = list(node.decorator_list) + a.defaults + [k for k in a.kw_defaults if k is not None]
        pre += [arg.annotation for arg in a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]
                if arg.annotation]
        if node.returns:
            pre.append(node.returns)
        self._visit_scoped(node, pre, node.body)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        a = node.args
        self._visit_scoped(node, a.defaults + [k for k in a.kw_defaults if k is not None], [node.body])

    def visit_ClassDef(self, node):
        if self.scope.parent is None and node.name in self.names:
            self._def_name(node, "class")
        self._visit_scoped(node, node.decorator_list + node.bases + [k.value for k in node.keywords], node.body)

    def _visit_comp(self, node, elts):
        self.visit(node.generators[0].iter)
        outer, self.scope = self.scope, node._scope
        for i, g in enumerate(node.generators):
            if i:
                self.visit(g.iter)
            self.visit(g.target)
            for c in g.ifs:
                self.visit(c)
        for e in elts:
            self.visit(e)
        self.scope = outer

    def visit_ListComp(self, node):
        self._visit_comp(node, [node.elt])

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node):
        self._visit_comp(node, [node.key, node.value])

    def visit_Name(self, node):
        if node.id in self.names and resolves_to_module(node.id, self.scope):
            assert node.end_lineno == node.lineno
            self._edit(node.lineno, node.col_offset, node.end_col_offset, node.id)

    def visit_Global(self, node):
        # `global a, b`: rename inside the statement's own text
        text = self.lines[node.lineno - 1]
        for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", text[node.col_offset:node.end_col_offset]):
            if m.group(0) in self.names:
                self._edit(node.lineno, node.col_offset + m.start(), node.col_offset + m.end(), m.group(0))


def module_names(tree):
    """Names bound at module level (defs, classes, assignments anywhere in module-level statements, `global` decls),
    excluding import aliases (shared imports are harmless and must not be renamed)."""
    b = Binder(tree)
    b.visit(tree)
    imported = set()
    for st in ast.walk(tree):
        if isinstance(st, (ast.Import, ast.ImportFrom)):
            for a in st.names:
                imported.add((a.asname or a.name).split(".")[0])
    return b.module.bound - imported


def prefix_module(src, prefix, keep=()):
    """Return src with every module-level name (except `keep`) prefixed."""
    tree = ast.parse(src)
    names = module_names(tree) - set(keep)
    lines = src.split("\n")
    r = Renamer(names, prefix, lines)
    r.visit_Module(tree)
    # apply edits from the end of each line backwards (ast columns are UTF-8 byte offsets)
    by_line = {}
    for ln, c0, c1, old, new in r.edits:
        by_line.setdefault(ln, []).append((c0, c1, old, new))
    for ln, eds in by_line.items():
        raw = lines[ln - 1].encode("utf-8")
        for c0, c1, old, new in sorted(set(eds), reverse=True):
            assert raw[c0:c1] == old.encode("utf-8"), (ln, c0, c1, old, raw[c0:c1])
            raw = raw[:c0] + new.encode("utf-8") + raw[c1:]
        lines[ln - 1] = raw.decode("utf-8")
    return "\n".join(lines), names


def strip_first_line(src):
    first, _, rest = src.partition("\n")
    assert first.startswith("# okc: problem="), first
    return rest


def free_names(src):
    """Names referenced as module-level bindings (anywhere) that nothing at module level, no import and no builtin
    provides."""
    tree = ast.parse(src)
    b = Binder(tree)
    b.visit(tree)
    defined = set(b.module.bound) | set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    missing = {}

    class V(ast.NodeVisitor):
        def __init__(self):
            self.scope = b.module

        def visit_Module(self, node):
            for st in node.body:
                self.visit(st)

        def _scoped(self, node, pre, body):
            for n in pre:
                self.visit(n)
            outer, self.scope = self.scope, node._scope
            for n in body:
                self.visit(n)
            self.scope = outer

        def visit_FunctionDef(self, node):
            a = node.args
            self._scoped(node, list(node.decorator_list) + a.defaults + [k for k in a.kw_defaults if k is not None], node.body)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, node):
            self._scoped(node, node.args.defaults, [node.body])

        def visit_ClassDef(self, node):
            self._scoped(node, node.decorator_list + node.bases, node.body)

        def _comp(self, node, elts):
            self.visit(node.generators[0].iter)
            outer, self.scope = self.scope, node._scope
            for i, g in enumerate(node.generators):
                if i:
                    self.visit(g.iter)
                for c in g.ifs:
                    self.visit(c)
            for e in elts:
                self.visit(e)
            self.scope = outer

        def visit_ListComp(self, node):
            self._comp(node, [node.elt])

        visit_SetComp = visit_ListComp
        visit_GeneratorExp = visit_ListComp

        def visit_DictComp(self, node):
            self._comp(node, [node.key, node.value])

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id not in defined and resolves_to_module(node.id, self.scope):
                missing.setdefault(node.id, node.lineno)

    V().visit_Module(tree)
    return missing


# ----------------------------------------------------------------------------------------------- composition
def compose(attn, moe, proj, glue, out, check):
    glue_src = pathlib.Path(glue).read_text()
    parts = {"attn": ("att_", attn), "moe": ("moe_", moe), "proj": ("bfp_", proj)}
    all_names = {}
    preexisting = {}   # prefixed name -> component: unresolved inside the component itself (its own dead code)
    for tag, (prefix, path) in parts.items():
        marker = "# {{COMPONENT %s}}" % tag
        assert marker in glue_src, marker
        src = strip_first_line(pathlib.Path(path).read_text())
        for k in free_names(src):   # an unbound name is never prefixed, so it keeps its spelling in the output
            preexisting[k] = tag
        body, names = prefix_module(src, prefix)
        all_names[tag] = names
        rel = pathlib.Path(path).resolve()
        try:
            rel = rel.relative_to(ROOT)
        except ValueError:
            pass
        banner = ("# " + "=" * 110 + "\n"
                  f"# component: {tag} ({rel}); every module-level name of that file carries the prefix `{prefix}`\n"
                  "# " + "=" * 110 + "\n")
        glue_src = glue_src.replace(marker, banner + body.rstrip("\n") + "\n# ---- end of component: %s ----\n" % tag)
    assert glue_src.startswith("# okc: problem=muse_decode_layer\n"), glue_src[:60]
    pathlib.Path(out).write_text(glue_src)
    n = glue_src.count("\n")
    print(f"wrote {out}: {n} lines; prefixed names: " + ", ".join(f"{t} {len(v)}" for t, v in all_names.items()))
    if check:
        compile(glue_src, out, "exec")
        miss = free_names(glue_src)
        old = {k: v for k, v in miss.items() if k in preexisting}
        miss = {k: v for k, v in miss.items() if k not in preexisting}
        for k, v in sorted(old.items(), key=lambda kv: kv[1]):
            print(f"  note: {k} (L{v}) is unresolved inside the {preexisting[k]} component itself (its dead code)")
        if miss:
            print("UNRESOLVED module-level names (name: first line):")
            for k, v in sorted(miss.items(), key=lambda kv: kv[1]):
                print(f"  {k}: L{v}")
            return 1
        print("check ok: compiles; every module-level reference resolves")
    return 0


def main(argv):
    if argv and argv[0] == "unprefix":
        prefix, path, a, b = argv[1], argv[2], int(argv[3]), int(argv[4])
        lines = pathlib.Path(path).read_text().split("\n")[a - 1:b]
        sys.stdout.write(re.sub(rf"\b{re.escape(prefix)}", "", "\n".join(lines)) + "\n")
        return 0
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--attn", default=str(ROOT / "muse_attention_decode/submission.py"))
    p.add_argument("--moe", default=str(ROOT / "muse_moe_w4a16/submission.py"))
    p.add_argument("--proj", default=str(ROOT / "muse_decode_layer/proj_fork.py"))
    p.add_argument("--glue", default=str(ROOT / "muse_decode_layer/layer_glue.py"))
    p.add_argument("-o", "--out", default=str(ROOT / "muse_decode_layer/submission.py"))
    p.add_argument("--check", action="store_true", help="compile the result and verify every module-level reference")
    a = p.parse_args(argv)
    return compose(a.attn, a.moe, a.proj, a.glue, a.out, a.check)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
