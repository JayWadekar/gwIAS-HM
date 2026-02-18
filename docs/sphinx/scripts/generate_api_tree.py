#!/usr/bin/env python3
"""Generate static, tree-structured API docs from Pipeline source using AST."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_DIR = REPO_ROOT / "Pipeline"
TREE_DIR = REPO_ROOT / "docs" / "sphinx" / "source" / "api" / "tree"

MODULE_PURPOSES: Dict[str, str] = {
    "triggers_single_detector_HM": "Single-detector triggering engine and trigger lifecycle.",
    "data_operations": "PSD estimation, whitening, and glitch/line mitigation utilities.",
    "coincidence_HM": "Multi-detector coincidence, vetoing, and candidate output logic.",
    "coherent_score_hm_search": "Coherent score marginalization routines for HM searches.",
    "coherent_score_mz_fast": "Fast and legacy coherent-scoring support utilities.",
    "ranking_HM": "Candidate aggregation, reweighting, and ranking logic.",
    "template_bank_generator_HM": "HM template-bank generation and basis-coefficient utilities.",
    "template_bank_params_O3a_HM": "Template-bank hyperparameter settings for observing runs.",
    "triggering_on_cluster": "Cluster submission and trigger-run orchestration helpers.",
    "gw_detect_file": "CLI entry points for per-file single-detector trigger generation.",
    "utils": "Cross-cutting utility functions and run/path helpers.",
    "python_utils": "Numeric and Python utility helpers shared across modules.",
    "readligo": "LIGO frame/HDF5 loading and segment-manipulation helpers.",
    "download_data": "GWOSC data download helpers.",
    "ML_modules": "Optional ML prior/posterior helpers used by ranking/scoring stages.",
    "params": "Global constants and threshold parameters used across the pipeline.",
}


@dataclass
class ParamDoc:
    type_text: str = "-"
    description: str = "-"


@dataclass
class ReturnDoc:
    type_text: str = "-"
    description: str = "-"


@dataclass
class FuncInfo:
    name: str
    signature: str
    summary: str
    params: List[Dict[str, str]]
    param_docs: Dict[str, ParamDoc] = field(default_factory=dict)
    return_annotation: str = "None"
    return_doc: ReturnDoc = field(default_factory=ReturnDoc)
    docstring: str = ""


@dataclass
class ClassInfo:
    name: str
    summary: str
    methods: List[FuncInfo]
    docstring: str = ""


@dataclass
class ModuleInfo:
    name: str
    summary: str
    purpose: str
    functions: List[FuncInfo]
    classes: List[ClassInfo]


def clean(text: Optional[str]) -> str:
    if not text:
        return "-"
    text = " ".join(text.strip().split())
    text = text.replace("\\", "\\\\")
    text = text.replace("|", "\\|")
    text = text.replace("`", "\\`")
    text = text.replace("*", "\\*")
    return text


def public_name(name: str) -> bool:
    return not name.startswith("_")


def first_sentence(docstring: str) -> str:
    if not docstring:
        return "No module-level description available."
    paragraph = docstring.strip().split("\n\n", 1)[0].strip()
    paragraph = " ".join(paragraph.split())
    cut_points = []
    for marker in [":param", ":return", " Parameters ", " Args ", " Returns ", " Yields "]:
        idx = paragraph.find(marker)
        if idx != -1:
            cut_points.append(idx)
    if cut_points:
        paragraph = paragraph[: min(cut_points)].strip()
    return clean(paragraph)


def format_arg(arg: ast.arg, default: Optional[ast.expr], prefix: str = "") -> Dict[str, str]:
    annotation = ast.unparse(arg.annotation) if arg.annotation is not None else "-"
    default_text = ast.unparse(default) if default is not None else "-"
    token = f"{prefix}{arg.arg}"
    if arg.annotation is not None:
        token += f": {annotation}"
    if default is not None:
        token += f" = {default_text}"
    return {
        "name": f"{prefix}{arg.arg}",
        "annotation": annotation,
        "default": default_text,
        "token": token,
    }


def build_signature_and_params(node: ast.FunctionDef | ast.AsyncFunctionDef, is_method: bool) -> tuple[str, List[Dict[str, str]], str]:
    a = node.args
    parts: List[str] = []
    params: List[Dict[str, str]] = []

    positional = list(a.posonlyargs) + list(a.args)
    defaults = [None] * (len(positional) - len(a.defaults)) + list(a.defaults)

    if a.posonlyargs:
        for idx, arg in enumerate(a.posonlyargs):
            entry = format_arg(arg, defaults[idx])
            parts.append(entry["token"])
            params.append(entry)
        parts.append("/")

    for idx, arg in enumerate(a.args):
        default = defaults[len(a.posonlyargs) + idx]
        entry = format_arg(arg, default)
        parts.append(entry["token"])
        params.append(entry)

    if a.vararg is not None:
        entry = format_arg(a.vararg, None, prefix="*")
        parts.append(entry["token"])
        params.append(entry)
    elif a.kwonlyargs:
        parts.append("*")

    for idx, arg in enumerate(a.kwonlyargs):
        default = a.kw_defaults[idx]
        entry = format_arg(arg, default)
        parts.append(entry["token"])
        params.append(entry)

    if a.kwarg is not None:
        entry = format_arg(a.kwarg, None, prefix="**")
        parts.append(entry["token"])
        params.append(entry)

    if is_method:
        params = [p for p in params if p["name"] not in {"self", "cls"}]

    return_annotation = ast.unparse(node.returns) if node.returns is not None else "None"
    sig = f"{node.name}({', '.join(parts)})"
    if node.returns is not None:
        sig += f" -> {return_annotation}"
    return sig, params, return_annotation


def extract_sections(docstring: str) -> Dict[str, str]:
    lines = docstring.splitlines()
    sections: Dict[str, str] = {}
    i = 0
    while i < len(lines) - 1:
        title = lines[i].strip()
        underline = lines[i + 1].strip()
        if title and set(underline) == {"-"} and len(underline) >= len(title):
            j = i + 2
            block: List[str] = []
            while j < len(lines):
                if j + 1 < len(lines):
                    next_title = lines[j].strip()
                    next_underline = lines[j + 1].strip()
                    if next_title and set(next_underline) == {"-"} and len(next_underline) >= len(next_title):
                        break
                block.append(lines[j])
                j += 1
            sections[title.lower()] = "\n".join(block).strip()
            i = j
            continue
        i += 1
    return sections


def parse_param_docs(section_text: str) -> Dict[str, ParamDoc]:
    result: Dict[str, ParamDoc] = {}
    current_names: List[str] = []
    current_desc: List[str] = []

    def flush() -> None:
        if not current_names:
            return
        desc = clean(" ".join(current_desc)) if current_desc else "-"
        for n in current_names:
            existing = result.get(n, ParamDoc())
            if desc != "-":
                existing.description = desc
            result[n] = existing

    for raw in section_text.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped:
            continue
        if not line.startswith((" ", "\t")) and ":" in stripped:
            flush()
            left, right = stripped.split(":", 1)
            names = [x.strip() for x in left.split(",") if x.strip()]
            type_text = clean(right)
            current_names = names
            current_desc = []
            for n in names:
                result[n] = ParamDoc(type_text=type_text, description="-")
        elif current_names:
            current_desc.append(stripped)
    flush()
    return result


def parse_return_doc(section_text: str) -> ReturnDoc:
    lines = [ln.strip() for ln in section_text.splitlines() if ln.strip()]
    if not lines:
        return ReturnDoc()
    type_text = "-"
    description = "-"
    first = lines[0]
    if ":" in first:
        _, rhs = first.split(":", 1)
        type_text = clean(rhs)
        description = clean(" ".join(lines[1:])) if len(lines) > 1 else "-"
    else:
        type_text = clean(first)
        description = clean(" ".join(lines[1:])) if len(lines) > 1 else "-"
    return ReturnDoc(type_text=type_text, description=description)


def parse_sphinx_fields(docstring: str) -> tuple[Dict[str, ParamDoc], ReturnDoc]:
    param_docs: Dict[str, ParamDoc] = {}
    return_doc = ReturnDoc()
    lines = docstring.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith(":param "):
            payload = stripped[len(":param ") :]
            if ":" in payload:
                name, desc = payload.split(":", 1)
                name = name.strip()
                desc_lines = [desc.strip()]
                j = i + 1
                while j < len(lines):
                    nxt = lines[j]
                    if nxt.strip().startswith(":"):
                        break
                    if nxt.strip():
                        desc_lines.append(nxt.strip())
                    j += 1
                param_docs[name] = ParamDoc(type_text="-", description=clean(" ".join(desc_lines)))
                i = j
                continue
        if stripped.startswith(":return:") or stripped.startswith(":returns:"):
            marker = ":return:" if stripped.startswith(":return:") else ":returns:"
            desc = stripped[len(marker) :].strip()
            desc_lines = [desc] if desc else []
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if nxt.strip().startswith(":"):
                    break
                if nxt.strip():
                    desc_lines.append(nxt.strip())
                j += 1
            return_doc.description = clean(" ".join(desc_lines)) if desc_lines else "-"
            i = j
            continue
        if stripped.startswith(":rtype:"):
            return_doc.type_text = clean(stripped[len(":rtype:") :].strip())
        i += 1
    return param_docs, return_doc


def parse_function(node: ast.FunctionDef | ast.AsyncFunctionDef, is_method: bool) -> FuncInfo:
    docstring = ast.get_docstring(node) or ""
    sections = extract_sections(docstring)
    params_section = sections.get("parameters") or sections.get("args") or sections.get("arguments") or ""
    returns_section = sections.get("returns") or sections.get("yields") or ""
    sphinx_params, sphinx_return = parse_sphinx_fields(docstring)
    param_docs = parse_param_docs(params_section)
    for name, pdoc in sphinx_params.items():
        if name not in param_docs:
            param_docs[name] = pdoc
        elif param_docs[name].description == "-" and pdoc.description != "-":
            param_docs[name].description = pdoc.description
    return_doc = parse_return_doc(returns_section)
    if return_doc.description == "-" and sphinx_return.description != "-":
        return_doc.description = sphinx_return.description
    if return_doc.type_text == "-" and sphinx_return.type_text != "-":
        return_doc.type_text = sphinx_return.type_text
    signature, params, return_annotation = build_signature_and_params(node, is_method=is_method)
    summary = first_sentence(docstring) if docstring else "No docstring summary available."
    return FuncInfo(
        name=node.name,
        signature=signature,
        summary=summary,
        params=params,
        param_docs=param_docs,
        return_annotation=return_annotation,
        return_doc=return_doc,
        docstring=docstring,
    )


def parse_module(path: Path) -> ModuleInfo:
    module_name = path.stem
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    module_doc = ast.get_docstring(tree) or ""

    functions: List[FuncInfo] = []
    classes: List[ClassInfo] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and public_name(node.name):
            functions.append(parse_function(node, is_method=False))
        elif isinstance(node, ast.ClassDef) and public_name(node.name):
            methods: List[FuncInfo] = []
            seen_methods = set()
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and public_name(member.name):
                    if member.name in seen_methods:
                        continue
                    seen_methods.add(member.name)
                    methods.append(parse_function(member, is_method=True))
            classes.append(
                ClassInfo(
                    name=node.name,
                    summary=first_sentence(ast.get_docstring(node) or "") if ast.get_docstring(node) else "No class docstring summary available.",
                    methods=methods,
                    docstring=ast.get_docstring(node) or "",
                )
            )

    functions.sort(key=lambda f: f.name.lower())
    classes.sort(key=lambda c: c.name.lower())

    purpose = MODULE_PURPOSES.get(module_name, first_sentence(module_doc))
    summary = first_sentence(module_doc)
    return ModuleInfo(
        name=module_name,
        summary=summary,
        purpose=purpose,
        functions=functions,
        classes=classes,
    )


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def render_docstring_block(docstring: str) -> str:
    if not docstring:
        return "No docstring text available."
    lines = [".. code-block:: text", ""]
    for line in docstring.splitlines():
        lines.append(f"   {line}")
    return "\n".join(lines)


def render_callable_page(path: Path, title: str, summary: str, signature: str, params: List[Dict[str, str]], param_docs: Dict[str, ParamDoc], return_annotation: str, return_doc: ReturnDoc, docstring: str, back_link: str) -> None:
    rows: List[str] = []
    if params:
        rows.extend(
            [
                ".. list-table:: Input variables",
                "   :header-rows: 1",
                "",
                "   * - Name",
                "     - Type",
                "     - Default",
                "     - Description",
            ]
        )
        for param in params:
            name_key = param["name"].lstrip("*")
            pdoc = param_docs.get(name_key, ParamDoc())
            ptype = pdoc.type_text if pdoc.type_text != "-" else param["annotation"]
            rows.extend(
                [
                    f"   * - ``{clean(param['name'])}``",
                    f"     - {clean(ptype)}",
                    f"     - {clean(param['default'])}",
                    f"     - {clean(pdoc.description)}",
                ]
            )
    else:
        rows.append("This callable has no explicit input variables.")

    rows_block = "\n".join(rows)
    content = f"""
{title}
{'=' * len(title)}

Back to {back_link}

Summary
-------

{clean(summary)}

Signature
---------

.. code-block:: python

   def {signature}

{rows_block}

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``{clean(return_annotation)}``
     - {clean(return_doc.type_text)}
     - {clean(return_doc.description)}

Docstring
---------

{render_docstring_block(docstring)}
"""
    write(path, content)


def render_module_page(module: ModuleInfo) -> None:
    module_path = TREE_DIR / "modules" / f"{module.name}.rst"

    function_rows: List[str] = []
    if module.functions:
        function_rows.extend(
            [
                ".. list-table:: Top-level functions",
                "   :header-rows: 1",
                "",
                "   * - Function",
                "     - Summary",
            ]
        )
        for fn in module.functions:
            fn_doc = f":doc:`{fn.name} <../functions/{module.name}.{fn.name}>`"
            function_rows.extend(
                [
                    f"   * - {fn_doc}",
                    f"     - {clean(fn.summary)}",
                ]
            )
    else:
        function_rows.append("No public top-level functions were found.")

    class_rows: List[str] = []
    if module.classes:
        class_rows.extend(
            [
                ".. list-table:: Classes",
                "   :header-rows: 1",
                "",
                "   * - Class",
                "     - Summary",
            ]
        )
        for cls in module.classes:
            cls_doc = f":doc:`{cls.name} <../classes/{module.name}.{cls.name}>`"
            class_rows.extend(
                [
                    f"   * - {cls_doc}",
                    f"     - {clean(cls.summary)}",
                ]
            )
    else:
        class_rows.append("No public classes were found.")

    child_docs: List[str] = []
    for cls in module.classes:
        child_docs.append(f"   ../classes/{module.name}.{cls.name}")
    for fn in module.functions:
        child_docs.append(f"   ../functions/{module.name}.{fn.name}")

    module_children = "\n".join(child_docs) if child_docs else "   "

    content = f"""
{module.name}
{'=' * len(module.name)}

Back to :doc:`API tree index <../index>`

Purpose
-------

{clean(module.purpose)}

Module summary
--------------

{clean(module.summary)}

{"\n".join(function_rows)}

{"\n".join(class_rows)}

.. toctree::
   :hidden:
   :maxdepth: 1

{module_children}
"""
    write(module_path, content)


def render_class_page(module_name: str, cls: ClassInfo) -> None:
    class_title = f"{module_name}.{cls.name}"
    class_path = TREE_DIR / "classes" / f"{module_name}.{cls.name}.rst"

    method_rows: List[str] = []
    if cls.methods:
        method_rows.extend(
            [
                ".. list-table:: Methods",
                "   :header-rows: 1",
                "",
                "   * - Method",
                "     - Summary",
            ]
        )
        for method in cls.methods:
            method_doc = f":doc:`{method.name} <../methods/{module_name}.{cls.name}.{method.name}>`"
            method_rows.extend(
                [
                    f"   * - {method_doc}",
                    f"     - {clean(method.summary)}",
                ]
            )
    else:
        method_rows.append("No public methods were found.")

    method_docs = "\n".join(
        f"   ../methods/{module_name}.{cls.name}.{method.name}" for method in cls.methods
    )
    class_children = method_docs if method_docs else "   "

    content = f"""
{class_title}
{'=' * len(class_title)}

Back to :doc:`Module page <../modules/{module_name}>`

Summary
-------

{clean(cls.summary)}

{"\n".join(method_rows)}

Class docstring
---------------

{render_docstring_block(cls.docstring)}

.. toctree::
   :hidden:
   :maxdepth: 1

{class_children}
"""
    write(class_path, content)


def render_tree_index(modules: List[ModuleInfo]) -> None:
    rows = [
        ".. list-table:: Modules",
        "   :header-rows: 1",
        "",
        "   * - Module",
        "     - Purpose",
    ]
    for module in modules:
        module_doc = f":doc:`{module.name} <modules/{module.name}>`"
        rows.extend(
            [
                f"   * - {module_doc}",
                f"     - {clean(module.purpose)}",
            ]
        )

    module_docs = "\n".join(f"   modules/{module.name}" for module in modules)

    content = f"""
API Tree
========

This API is organized as a navigation tree:

1. Module pages
2. Per-module tables for top-level functions and classes
3. Per-class tables for methods
4. Per-callable pages with signature, input variables, and output variables

{"\n".join(rows)}

.. toctree::
   :hidden:
   :maxdepth: 1

{module_docs}
"""
    write(TREE_DIR / "index.rst", content)


def main() -> None:
    modules: List[ModuleInfo] = []
    for path in sorted(PIPELINE_DIR.glob("*.py")):
        modules.append(parse_module(path))

    for module in modules:
        render_module_page(module)

        for fn in module.functions:
            render_callable_page(
                path=TREE_DIR / "functions" / f"{module.name}.{fn.name}.rst",
                title=f"{module.name}.{fn.name}",
                summary=fn.summary,
                signature=fn.signature,
                params=fn.params,
                param_docs=fn.param_docs,
                return_annotation=fn.return_annotation,
                return_doc=fn.return_doc,
                docstring=fn.docstring,
                back_link=f":doc:`Module page <../modules/{module.name}>`",
            )

        for cls in module.classes:
            render_class_page(module.name, cls)
            for method in cls.methods:
                render_callable_page(
                    path=TREE_DIR / "methods" / f"{module.name}.{cls.name}.{method.name}.rst",
                    title=f"{module.name}.{cls.name}.{method.name}",
                    summary=method.summary,
                    signature=method.signature,
                    params=method.params,
                    param_docs=method.param_docs,
                    return_annotation=method.return_annotation,
                    return_doc=method.return_doc,
                    docstring=method.docstring,
                    back_link=f":doc:`Class page <../classes/{module.name}.{cls.name}>`",
                )

    render_tree_index(modules)


if __name__ == "__main__":
    main()
