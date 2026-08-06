# 📚 Library Catalog Q&A (Streamlit + pandas + ChatGPT fallback)

A cost-effective chatbot for asking natural-language questions about a book
catalog. Most questions are answered **instantly and for free** using pandas
against the spreadsheet; only questions the rule-based engine can't handle
fall back to OpenAI's `gpt-4o-mini`.

## How it keeps costs low

1. **Load once, cache forever.** The catalog is read from disk a single time
   per file version and cached with `st.cache_data` (`data_loader.py`).
2. **Pandas-first routing.** `rule_engine.py` recognizes ~15 common question
   shapes (counts, filters by author/category/year, oldest/newest, duplicate
   & review flags, edition counts, title lookups, etc.) with regex + fuzzy
   matching and answers them directly — **zero LLM tokens spent.**
3. **Cheap, minimal LLM fallback.** Only when nothing matches does
   `llm_engine.py` call OpenAI — and even then it doesn't send any row data.
   It sends just the column schema + 3 sample rows and asks for a single
   pandas expression, which is executed locally and formatted the same way
   as the fast path. That keeps each fallback call to a few hundred tokens.
4. **Visible cost tracking.** The sidebar shows how many questions were
   answered via pandas vs. the LLM, and each answer is tagged with its source
   and latency.

## Files

```
app.py              Streamlit UI / chat loop
data_loader.py       Cached catalog loading (.xlsx / .ods / .csv)
rule_engine.py        Pandas-based "fast path" question answering
llm_engine.py          OpenAI fallback: generates + safely executes pandas code
data/sample_catalog.ods  Example data file (your own katalog.xlsx works too)
requirements.txt
.streamlit/secrets.toml.example
```

## Expected columns

```
Book_ID, Title, Author, Category, Category_NonLatin_Flag, Year_Field_Raw,
Number_of_Editions_Recorded, Earliest_Year, Latest_Year, Possible_Duplicate,
Needs_Review
```

`Category_NonLatin_Flag`, `Possible_Duplicate`, and `Needs_Review` are
treated as booleans (1/blank in the sheet → True/False).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

By default it loads `data/sample_catalog.ods`. To point it at your own file
without re-uploading through the UI every time, set:

```bash
export CATALOG_PATH=/path/to/katalog.xlsx
streamlit run app.py
```

You can also just use the "Replace catalog file" uploader in the sidebar at
runtime (accepts `.xlsx`, `.ods`, `.csv`).

## OpenAI API key

Only needed for the LLM fallback path — the app works without one, it will
just tell you it can't answer questions outside the rule engine's coverage.

- **Locally / no secrets file:** paste it into the sidebar field, or
  `export OPENAI_API_KEY=sk-...` before running.
- **Streamlit Community Cloud:** copy `.streamlit/secrets.toml.example` to
  `.streamlit/secrets.toml` (kept out of git) or paste its contents into the
  app's *Settings → Secrets* box in the Streamlit Cloud dashboard:

  ```toml
  OPENAI_API_KEY = "sk-..."
  ```

The model used for fallback (`gpt-4o-mini` by default — OpenAI's cheapest
capable chat model) is selectable in the sidebar.

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at `app.py`.
3. Add `OPENAI_API_KEY` under app Settings → Secrets.
4. (Optional) Commit your real catalog file under `data/` and set
   `CATALOG_PATH` accordingly, or just upload it via the sidebar after the
   app is live.

## Extending the rule engine

If you find yourself repeatedly asking the same kind of question and it's
going to the (paid) LLM fallback, add a small regex handler for it in
`rule_engine.py::answer()` — that's the main lever for keeping costs near
zero as usage grows.

## Safety notes on the LLM fallback

The fallback never lets the model run arbitrary code against your data: the
generated snippet is parsed with Python's `ast` module and rejected if it
contains imports, dunder attribute access, `exec`/`eval`/`open`, or anything
other than a single expression. Only a small allow-list of builtins (`len`,
`sum`, `min`, `max`, `sorted`, etc.) plus `pandas`/`df` are available when it
runs.
