"""
LLM fallback engine.

"""
import ast
import os
import re
import textwrap

import pandas as pd
from openai import OpenAI

from rule_engine import _fmt_table  # reuse the same table formatting

SYSTEM_PROMPT = """You are a code generator for a pandas DataFrame called `df` \
that holds a library catalog. Given a user's natural-language question, output \
ONLY a single Python expression (no assignment, no imports, no comments, no \
markdown fences, no explanation) that evaluates to the answer using `df`. \
The expression may span multiple statements only if wrapped as a single \
expression using a lambda or comprehension - prefer a single clear expression. \
If a print/explanation is needed, still return only a valid Python expression \
whose value is the answer (a DataFrame, Series, list, number, or string). \
Never use import, open, exec, eval, os, sys, or any dunder attribute."""

FORBIDDEN_NODE_TYPES = (
    ast.Import, ast.ImportFrom, ast.With, ast.AsyncWith,
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
    ast.Global, ast.Nonlocal, ast.Delete,
)


def _schema_context(df: pd.DataFrame) -> str:
    categories = sorted(df["Category"].dropna().unique().tolist())
    authors = sorted(df["Author"].dropna().unique().tolist())
    sample = df.drop(columns=[c for c in df.columns if c.startswith("_")]).head(3).to_dict(orient="records")
    return textwrap.dedent(f"""
        Columns: {list(df.columns)}
        Categories ({len(categories)}): {categories}
        Number of authors: {len(authors)}
        Sample rows: {sample}
        Notes:
        - Category_NonLatin_Flag, Possible_Duplicate, Needs_Review are booleans.
        - Earliest_Year / Latest_Year / Number_of_Editions_Recorded are numeric.
        - Year_Field_Raw is the original messy text (e.g. "1908, 1914 (2nd ed.)").
    """).strip()


def _extract_code(raw: str) -> str:
    code = raw.strip()
    code = re.sub(r"^```(python)?", "", code).strip()
    code = re.sub(r"```$", "", code).strip()
    return code


def _is_safe_expression(code: str) -> bool:
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODE_TYPES):
            return False
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False
        if isinstance(node, ast.Name) and node.id in {"exec", "eval", "open", "os", "sys", "__import__"}:
            return False
    return True


def _run_expression(code: str, df: pd.DataFrame):
    safe_globals = {"__builtins__": {
        "len": len, "sum": sum, "min": min, "max": max, "sorted": sorted,
        "round": round, "str": str, "int": int, "float": float,
        "list": list, "dict": dict, "range": range, "abs": abs,
    }}
    safe_locals = {"df": df, "pd": pd}
    return eval(code, safe_globals, safe_locals)  # noqa: S307 (validated by _is_safe_expression)


def _format_result(result) -> str:
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return "No matching books found."
        return f"**{len(result)} result(s)**:\n\n" + _fmt_table(result, cols=list(result.columns))
    if isinstance(result, pd.Series):
        if result.empty:
            return "No matching results."
        df_ = result.reset_index()
        df_.columns = ["Item", "Value"]
        return df_.to_markdown(index=False)
    if isinstance(result, (list, tuple, set, pd.Index)) or hasattr(result, "tolist"):
        items = result.tolist() if hasattr(result, "tolist") else list(result)
        if len(items) == 0:
            return "No matching results."
        return ", ".join(str(i) for i in items)
    return str(result)


def answer(question: str, df: pd.DataFrame, api_key: str, model: str = "gpt-4o-mini") -> str:
    if not api_key:
        return ("I couldn't answer that from the structured data alone, and no OpenAI "
                "API key is configured, so I can't fall back to the LLM. Please add your "
                "API key in the sidebar.")

    client = OpenAI(api_key=api_key)
    context = _schema_context(df)

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=200,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{context}\n\nQuestion: {question}\n\nExpression:"},
            ],
        )
        code = _extract_code(resp.choices[0].message.content)
    except Exception as e:
        return f"LLM request failed: {e}"

    if not _is_safe_expression(code):
        return ("I generated a query I'm not confident is safe to run, so I stopped short. "
                "Try rephrasing the question.")

    try:
        result = _run_expression(code, df)
    except Exception as e:
        # One retry: tell the model what went wrong and ask for a fix.
        try:
            resp2 = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{context}\n\nQuestion: {question}\n\nExpression:"},
                    {"role": "assistant", "content": code},
                    {"role": "user", "content": f"That raised: {e}. Give a corrected single expression."},
                ],
            )
            code2 = _extract_code(resp2.choices[0].message.content)
            if not _is_safe_expression(code2):
                return "I couldn't safely answer that question - could you rephrase it?"
            result = _run_expression(code2, df)
        except Exception as e2:
            return f"I couldn't answer that question from the data ({e2})."

    return _format_result(result)
