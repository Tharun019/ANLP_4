"""
Topic Explorer - Streamlit front end for the ANLP_4 LDA pipeline.

Run:  streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import pipeline as P

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Topic Explorer - LDA", page_icon="🧩", layout="wide")


# ---------------------------------------------------------------------------
# Cached computations (re-run only when inputs change)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Fitting LDA ...")
def cached_run(texts, doc_ids, themes, k, top_n, seed, restarts, min_df, max_df):
    return P.run_pipeline(
        list(texts), n_topics=k, top_n=top_n, random_state=seed,
        doc_ids=list(doc_ids), true_themes=list(themes) if themes else None,
        min_df=min_df, max_df=max_df, n_restarts=restarts,
    )


@st.cache_data(show_spinner="Sweeping K = 2 ... 6 ...")
def cached_sweep(texts, themes, seed, top_n, min_df, max_df):
    cleaned, _, _ = P.preprocess(list(texts))
    X, vec = P.build_dtm(cleaned, min_df=min_df, max_df=max_df)
    max_k = min(6, X.shape[0], X.shape[1])
    return P.k_sweep(X, vec, ks=range(2, max_k + 1), random_state=seed,
                     true_themes=list(themes) if themes else None, top_n=top_n)


@st.cache_data(show_spinner="Re-fitting with 10 seeds ...")
def cached_stability(texts, themes, k, min_df, max_df):
    cleaned, _, _ = P.preprocess(list(texts))
    X, _ = P.build_dtm(cleaned, min_df=min_df, max_df=max_df)
    return P.seed_stability(X, k, list(themes))


# ---------------------------------------------------------------------------
# Sidebar - input and parameters
# ---------------------------------------------------------------------------

st.sidebar.title("🧩 Topic Explorer")
st.sidebar.caption("Latent Dirichlet Allocation on a small document collection")

source = st.sidebar.radio("Documents", ["Sample corpus (20 docs)", "Paste text", "Upload file"])

df = None
if source.startswith("Sample"):
    df = P.load_corpus(P.SAMPLE_CORPUS)
elif source == "Paste text":
    raw = st.sidebar.text_area("One document per line", height=220,
                               placeholder="The team won the match...\nThe stock market rose...")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if lines:
        df = P.corpus_from_texts(lines)
else:
    up = st.sidebar.file_uploader("CSV with a 'text' column, or TXT (one doc per line)", type=["csv", "txt"])
    if up is not None:
        if up.name.lower().endswith(".csv"):
            df = pd.read_csv(up)
            if "text" not in df.columns:
                st.sidebar.error("CSV needs a 'text' column.")
                df = None
            else:
                df = P._normalise_frame(df)
        else:
            lines = [ln for ln in up.read().decode("utf-8", "ignore").splitlines() if ln.strip()]
            df = P.corpus_from_texts(lines)

st.sidebar.divider()
k = st.sidebar.slider("Number of topics (K)", 2, 8, 4)
top_n = st.sidebar.slider("Top words per topic", 3, 10, 5)
restarts = st.sidebar.slider("LDA restarts (best log-likelihood kept)", 1, 20, 10)
seed = st.sidebar.number_input("Random seed", 0, 10_000, 42)
with st.sidebar.expander("Vocabulary filters"):
    min_df = st.number_input("min_df (min documents per word)", 1, 10, 2 if (df is not None and len(df) >= 10) else 1)
    max_df = st.slider("max_df (max fraction of documents)", 0.2, 1.0, 0.5, 0.05)

# ---------------------------------------------------------------------------
# Guard: need input
# ---------------------------------------------------------------------------

st.title("Topic Modeling with LDA")
st.caption("Discover hidden topics in a document collection and list the top words for each topic.")

if df is None or len(df) < 2:
    st.info("Add at least two documents in the sidebar to begin (or pick the sample corpus).")
    st.stop()

texts = tuple(df["text"])
doc_ids = tuple(df["doc_id"])
themes = tuple(df["true_theme"]) if "true_theme" in df.columns else None

try:
    res = cached_run(texts, doc_ids, themes, k, top_n, int(seed), restarts, int(min_df), float(max_df))
except ValueError as e:
    st.error(str(e))
    st.stop()

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------

m = st.columns(5 if res["agreement"] else 4)
m[0].metric("Documents", res["X"].shape[0])
m[1].metric("Vocabulary", res["X"].shape[1])
m[2].metric("Topics", k)
m[3].metric("Perplexity", f"{res['perplexity']:.1f}")
if res["agreement"]:
    m[4].metric("Purity vs themes", f"{res['agreement']['purity']:.2f}")

tabs = st.tabs(["Top words", "Documents", "Preprocessing", "Choosing K", "Evaluation", "Corpus", "Report"])

# ---- Tab 1: Top words (the main deliverable) --------------------------------
with tabs[0]:
    st.subheader(f"Top {top_n} words for each discovered topic")
    st.dataframe(res["top_words_wide"], hide_index=True, width="stretch")

    tw = res["top_words"]
    ncols = min(k, 4)
    nrows = -(-k // ncols)
    labels = res["labels"] or {}
    titles = [f"Topic {t + 1}" + (f" - {labels[t + 1]}" if labels.get(t + 1) else "") for t in range(k)]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles, horizontal_spacing=0.08, vertical_spacing=0.18)
    for t in range(k):
        d = tw[tw["topic"] == t].sort_values("weight")
        fig.add_trace(go.Bar(x=d["weight"], y=d["word"], orientation="h", showlegend=False,
                             hovertemplate="%{y}: %{x:.3f}<extra></extra>"),
                      row=t // ncols + 1, col=t % ncols + 1)
    fig.update_layout(height=260 * nrows, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(title_text="P(word | topic)", title_font_size=10)
    st.plotly_chart(fig, width="stretch")
    st.caption("Weight = P(word | topic): each row of lda.components_ normalised to sum to 1.")

    c1, c2 = st.columns(2)
    c1.download_button("Download top words (CSV)", res["top_words_wide"].to_csv(index=False),
                       "topics_top_words.csv", "text/csv")
    c2.download_button("Download word weights (CSV)", res["top_words"].to_csv(index=False),
                       "topic_word_weights.csv", "text/csv")

# ---- Tab 2: Documents --------------------------------------------------------
with tabs[1]:
    st.subheader("Topic mixture of each document")
    dt = res["doc_topics"]
    topic_cols = [c for c in dt.columns if c.startswith("Topic ")]
    heat = px.imshow(dt.set_index("doc_id")[topic_cols], color_continuous_scale="Blues",
                     zmin=0, zmax=1, aspect="auto", labels=dict(color="P(topic | doc)"))
    heat.update_layout(height=max(320, 24 * len(dt)), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(heat, width="stretch")

    view = dt.copy()
    view.insert(1, "text", [t[:90] + ("..." if len(t) > 90 else "") for t in df["text"]])
    if themes:
        view.insert(2, "true_theme", themes)
    st.dataframe(view.round(3), hide_index=True, width="stretch")
    st.download_button("Download document-topic table (CSV)", dt.round(4).to_csv(index=False),
                       "doc_topic_distribution.csv", "text/csv")

# ---- Tab 3: Preprocessing ----------------------------------------------------
with tabs[2]:
    st.subheader("From raw text to a document-term matrix")
    pick = st.selectbox("Document", doc_ids)
    i = doc_ids.index(pick)
    a, b = st.columns(2)
    a.markdown("**Raw text**")
    a.write(texts[i])
    b.markdown("**After lowercasing, stopword removal and plural merging**")
    b.code(res["cleaned_texts"][i], language=None)

    st.markdown(f"**Document-term matrix:** {res['X'].shape[0]} documents x {res['X'].shape[1]} words "
                f"(min_df = {int(min_df)}, max_df = {max_df:.2f})")
    freq = pd.DataFrame({"word": res["feature_names"], "count": res["X"].sum(axis=0).A1})
    freq = freq.sort_values("count", ascending=False).head(25)
    st.plotly_chart(px.bar(freq, x="word", y="count", height=300).update_layout(margin=dict(t=10, b=10)),
                    width="stretch")
    if res["plural_map"]:
        st.markdown("**Plural forms merged** (only when the singular also appears in the corpus)")
        st.dataframe(pd.DataFrame(sorted(res["plural_map"].items()), columns=["plural", "merged into"]),
                     hide_index=True)

# ---- Tab 4: Choosing K -------------------------------------------------------
with tabs[3]:
    st.subheader("How many topics?")
    st.write("Each K is fitted with restarts. Lower perplexity = the model explains the words better; "
             "higher UMass coherence = a topic's top words appear together more often.")
    sweep = cached_sweep(texts, themes, int(seed), top_n, int(min_df), float(max_df))
    best_k = int(sweep.loc[sweep["perplexity"].idxmin(), "k"])
    cols = ["perplexity", "mean_umass"] + (["purity"] if "purity" in sweep else [])
    fig = make_subplots(rows=1, cols=len(cols), subplot_titles=[c.replace("_", " ") for c in cols])
    for j, c in enumerate(cols, start=1):
        fig.add_trace(go.Scatter(x=sweep["k"], y=sweep[c], mode="lines+markers", showlegend=False), row=1, col=j)
        fig.add_vline(x=best_k, line_dash="dash", line_color="orange", row=1, col=j)
    fig.update_xaxes(dtick=1, title_text="K")
    fig.update_layout(height=320, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(sweep.round(3), hide_index=True)
    st.success(f"Lowest perplexity at K = {best_k}.")

# ---- Tab 5: Evaluation -------------------------------------------------------
with tabs[4]:
    st.subheader("How good are the topics?")
    c1, c2 = st.columns(2)
    c1.markdown("**UMass coherence per topic** (closer to 0 = more coherent)")
    c1.dataframe(res["coherence"].round(3), hide_index=True)
    c2.markdown("**Run details**")
    c2.write({
        "EM iterations (winning run)": res["n_iter"],
        "Winning restart seed": res["best_seed"],
        "Approx. log-likelihood": round(res["loglik"], 2),
    })
    if res["agreement"]:
        a = res["agreement"]
        st.markdown("**Agreement with the known themes** (LDA never saw these labels)")
        e1, e2, e3 = st.columns(3)
        e1.metric("Purity", f"{a['purity']:.3f}")
        e2.metric("NMI", f"{a['nmi']:.3f}")
        e3.metric("ARI", f"{a['ari']:.3f}")
        ct = a["contingency"].copy()
        ct.index = [f"Topic {t}" for t in ct.index]
        st.dataframe(ct, width="stretch")
        if st.checkbox("Run seed-stability check (single run vs best of 10 restarts)"):
            stab = cached_stability(texts, themes, k, int(min_df), float(max_df))
            st.dataframe(stab.round(3), hide_index=True)
            st.write(stab.iloc[:, 1:].mean().round(3).to_dict())
    else:
        st.info("Upload a CSV with a 'true_theme' column to compare topics with known labels.")

# ---- Tab 6: Corpus -----------------------------------------------------------
with tabs[5]:
    st.subheader("Input documents")
    st.dataframe(df, hide_index=True, width="stretch")

# ---- Tab 7: Report -----------------------------------------------------------
with tabs[6]:
    readme = ROOT / "README.md"
    st.markdown(readme.read_text() if readme.exists() else "README.md not found.")
