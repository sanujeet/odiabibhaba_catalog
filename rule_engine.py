"""
"""
import re
import pandas as pd
from rapidfuzz import process, fuzz

MAX_LIST_ROWS = 25  # keep chat replies readable; mention truncation if exceeded


def _fmt_table(df: pd.DataFrame, cols=None) -> str:
    if cols is None:
        cols = ["Title", "Author", "Category", "Earliest_Year", "Latest_Year"]
    cols = [c for c in cols if c in df.columns]
    view = df[cols].copy()
    truncated = len(view) > MAX_LIST_ROWS
    if truncated:
        view = view.head(MAX_LIST_ROWS)
    md = view.to_markdown(index=False)
    if truncated:
        md += f"\n\n_...and {len(df) - MAX_LIST_ROWS} more (showing first {MAX_LIST_ROWS})._"
    return md


def _extract_category(q: str, categories: list[str]) -> str | None:
    ql = q.lower()
    # Whole-word match, longest category names first, so a category like
    # "Poet" can't accidentally swallow a question about "Poetry" (it was
    # matching as a plain substring before, which is the bug being fixed here).
    for c in sorted(categories, key=len, reverse=True):
        if re.search(rf"\b{re.escape(c.lower())}\b", ql):
            return c
    # fuzzy fallback for typos, e.g. "poetary"
    match = process.extractOne(ql, categories, scorer=fuzz.partial_ratio, score_cutoff=85)
    return match[0] if match else None


def _extract_author(q: str, authors: list[str]) -> str | None:
    match = process.extractOne(q, authors, scorer=fuzz.token_set_ratio, score_cutoff=70)
    return match[0] if match else None


def _extract_title(q: str, titles: list[str]) -> str | None:
    # partial_ratio finds the title as a near-exact substring of the question,
    # which is robust to surrounding words ("tell me about X?", "who wrote X?")
    # while staying discriminative enough not to fire on unrelated questions.
    match = process.extractOne(q, titles, scorer=fuzz.partial_ratio, score_cutoff=90)
    return match[0] if match else None


def _extract_years(q: str) -> list[int]:
    return [int(y) for y in re.findall(r"\b(1[5-9]\d{2}|20\d{2})\b", q)]


def answer(question: str, df: pd.DataFrame) -> str | None:
    q = question.strip()
    ql = q.lower()
    categories = sorted(df["Category"].dropna().unique().tolist())
    authors = sorted(df["Author"].dropna().unique().tolist())
    titles = df["Title"].dropna().unique().tolist()
    years = _extract_years(q)

    # ---- 1. total count of books --------------------------------------
    if re.search(r"\bhow many books\b|\btotal (number of )?books\b|\bcount (of )?books\b", ql):
        cat = _extract_category(q, categories)
        auth = _extract_author(q, authors) if " by " in ql or "author" in ql else None
        sub = df
        label = "books"
        if cat:
            sub = sub[sub["Category"] == cat]
            label += f" in '{cat}'"
        if auth:
            sub = sub[sub["Author"] == auth]
            label += f" by {auth}"
        if years:
            y = years[0]
            sub = sub[(sub["Earliest_Year"] <= y) & (sub["Latest_Year"] >= y)]
            label += f" (active in {y})"
        return f"There are **{len(sub)}** {label}."

    # ---- 2. list unique categories / authors ---------------------------
    if re.search(r"\bwhat categories\b|\blist (all )?categories\b|\bwhich categories\b", ql):
        return "Categories in the catalog:\n\n" + "\n".join(f"- {c}" for c in categories)

    if re.search(r"\blist (all )?authors\b|\bwhich authors\b|\bwho are the authors\b", ql):
        shown, extra = authors[:MAX_LIST_ROWS], len(authors) - MAX_LIST_ROWS
        text = "Authors in the catalog:\n\n" + "\n".join(f"- {a}" for a in shown)
        if extra > 0:
            text += f"\n\n_...and {extra} more ({len(authors)} total)._"
        return text

    # ---- 3. books by a specific author ---------------------------------
    m = re.search(r"books?\s+(?:by|from)\s+(.+?)(?:\?|$)", ql)
    if m or ("author" in ql and any(a.lower() in ql for a in authors)):
        auth = _extract_author(m.group(1) if m else q, authors)
        if auth:
            sub = df[df["Author"] == auth]
            return f"**{len(sub)} book(s)** by {auth}:\n\n" + _fmt_table(sub)

    # ---- 4. books in a specific category --------------------------------
    if re.search(r"\bcategory\b|\bgenre\b", ql) or _extract_category(q, categories):
        cat = _extract_category(q, categories)
        if cat and re.search(r"\blist\b|\bshow\b|\bwhich books\b|\bwhat books\b", ql):
            sub = df[df["Category"] == cat]
            return f"**{len(sub)} book(s)** in category '{cat}':\n\n" + _fmt_table(sub)

    # ---- 5. year range / before / after ----------------------------------
    if re.search(r"\bbefore\b", ql) and years:
        y = years[0]
        sub = df[df["Latest_Year"] < y]
        return f"**{len(sub)} book(s)** published before {y}:\n\n" + _fmt_table(sub)

    if re.search(r"\bafter\b", ql) and years:
        y = years[0]
        sub = df[df["Earliest_Year"] > y]
        return f"**{len(sub)} book(s)** published after {y}:\n\n" + _fmt_table(sub)

    if re.search(r"\bbetween\b", ql) and len(years) >= 2:
        y1, y2 = sorted(years[:2])
        sub = df[(df["Earliest_Year"] >= y1) & (df["Latest_Year"] <= y2)]
        return f"**{len(sub)} book(s)** published between {y1} and {y2}:\n\n" + _fmt_table(sub)

    if len(years) == 1 and re.search(r"\bin (year )?\d{4}\b|\bpublished in\b", ql):
        y = years[0]
        sub = df[(df["Earliest_Year"] <= y) & (df["Latest_Year"] >= y)]
        return f"**{len(sub)} book(s)** with an edition in {y}:\n\n" + _fmt_table(sub)

    # ---- 6. oldest / newest book -----------------------------------------
    if re.search(r"\boldest\b|\bearliest\b", ql):
        sub = df.nsmallest(5, "Earliest_Year") if "s" in ql[:20] or "top" in ql or "5" in ql else df.loc[[df["Earliest_Year"].idxmin()]]
        return "Oldest book(s) in the catalog:\n\n" + _fmt_table(sub)

    if re.search(r"\bnewest\b|\blatest\b|\bmost recent\b", ql):
        sub = df.nlargest(5, "Latest_Year") if "s" in ql[:20] or "top" in ql or "5" in ql else df.loc[[df["Latest_Year"].idxmax()]]
        return "Newest / most recent book(s) in the catalog:\n\n" + _fmt_table(sub)

    # ---- 7. duplicates / needs review flags -------------------------------
    if "duplicate" in ql:
        sub = df[df["Possible_Duplicate"] == True]  # noqa: E712
        if sub.empty:
            return "No books are currently flagged as possible duplicates."
        return f"**{len(sub)} book(s)** flagged as possible duplicates:\n\n" + _fmt_table(sub)

    if "needs review" in ql or "need review" in ql or "flagged" in ql:
        sub = df[df["Needs_Review"] == True]  # noqa: E712
        if sub.empty:
            return "No books are currently flagged as needing review."
        return f"**{len(sub)} book(s)** flagged as needing review:\n\n" + _fmt_table(sub)

    if "non-latin" in ql or "nonlatin" in ql or "non latin" in ql:
        sub = df[df["Category_NonLatin_Flag"] == True]  # noqa: E712
        if sub.empty:
            return "No books are flagged with a non-Latin category script."
        return f"**{len(sub)} book(s)** flagged non-Latin category:\n\n" + _fmt_table(sub)

    # ---- 8. multiple editions ----------------------------------------------
    if "multiple editions" in ql or "more than one edition" in ql or re.search(r"\bediti", ql):
        if re.search(r"\bhow many editions\b", ql):
            title = _extract_title(q, titles)
            if title:
                row = df[df["Title"] == title].iloc[0]
                return (f"**{title}** by {row['Author']} has "
                        f"**{row['Number_of_Editions_Recorded']}** recorded edition(s) "
                        f"({row['Earliest_Year']}\u2013{row['Latest_Year']}).")
        if "multiple" in ql or "more than one" in ql:
            sub = df[df["Number_of_Editions_Recorded"] > 1]
            return f"**{len(sub)} book(s)** with multiple recorded editions:\n\n" + _fmt_table(
                sub, ["Title", "Author", "Number_of_Editions_Recorded", "Earliest_Year", "Latest_Year"]
            )

    # ---- 9. lookup a specific title (details) -------------------------------
    title = _extract_title(q, titles)
    if title and re.search(r"\btell me about\b|\bdetails\b|\binfo\b|\bwho wrote\b|\bwhen was\b|\bshow me\b", ql):
        row = df[df["Title"] == title].iloc[0]
        return (
            f"**{row['Title']}**\n\n"
            f"- Author: {row['Author']}\n"
            f"- Category: {row['Category']}\n"
            f"- Editions recorded: {row['Number_of_Editions_Recorded']}\n"
            f"- Year range: {row['Earliest_Year']}\u2013{row['Latest_Year']} "
            f"(raw: {row['Year_Field_Raw']})\n"
            f"- Possible duplicate: {'Yes' if row['Possible_Duplicate'] else 'No'}\n"
            f"- Needs review: {'Yes' if row['Needs_Review'] else 'No'}"
        )

    # ---- 10. simple catalog-wide stats --------------------------------------
    if re.search(r"\bhow many categories\b", ql):
        return f"There are **{len(categories)}** distinct categories."
    if re.search(r"\bhow many authors\b", ql):
        return f"There are **{len(authors)}** distinct authors."
    if re.search(r"\baverage (number of )?editions\b", ql):
        return f"The average number of recorded editions per book is **{df['Number_of_Editions_Recorded'].mean():.2f}**."

    # No rule matched -> let the LLM handle it
    return None
