"""
ANLP_4 - Topic Modeling with Latent Dirichlet Allocation (LDA)
==============================================================

Problem statement:
    Apply LDA to a small collection of text documents and display the
    top five words associated with each discovered topic.

Pipeline stages (each one is a plain, inspectable function):
    1. load_corpus()          -> read documents (CSV / TXT / list of strings)
    2. preprocess()           -> lowercase, tokenize, remove stopwords,
                                 merge plural forms (players -> player)
    3. build_dtm()            -> bag-of-words document-term matrix (raw counts)
    4. fit_lda()              -> scikit-learn LatentDirichletAllocation
    5. top_words()            -> top-N words per topic from lda.components_
    6. doc_topic_table()      -> topic mixture of every document
    7. evaluation helpers     -> UMass coherence, perplexity, purity/NMI/ARI
                                 against known themes, K sweep, seed stability

run_pipeline(texts, n_topics) runs stages 2-6 and returns everything in one
dict, so the notebook, the Streamlit app and the CLI all get identical results.

CLI usage:
    python pipeline.py                      # sample corpus, K=4, saves to outputs/
    python pipeline.py --k 3 --top-n 5
    python pipeline.py --data my_docs.txt   # one document per line
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parent
SAMPLE_CORPUS = ROOT / "sample_data" / "corpus.csv"
OUTPUT_DIR = ROOT / "outputs"

# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------


def load_corpus(path: str | Path = SAMPLE_CORPUS) -> pd.DataFrame:
    """Load documents into a DataFrame with columns doc_id, text and
    (optionally) true_theme.

    * .csv -> must have a 'text' column; 'doc_id' / 'true_theme' are optional
    * .txt -> one document per non-empty line
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "text" not in df.columns:
            raise ValueError("CSV must contain a 'text' column")
    else:
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
        df = pd.DataFrame({"text": [ln for ln in lines if ln]})
    return _normalise_frame(df)


def corpus_from_texts(texts: list[str], themes: list[str] | None = None) -> pd.DataFrame:
    """Build the same DataFrame shape from a plain list of strings."""
    df = pd.DataFrame({"text": [t.strip() for t in texts if t and t.strip()]})
    if themes is not None and len(themes) == len(df):
        df["true_theme"] = themes
    return _normalise_frame(df)


def _normalise_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].astype(str)
    if "doc_id" not in df.columns:
        df.insert(0, "doc_id", [f"D{i + 1:02d}" for i in range(len(df))])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------

# Filler words that are not in scikit-learn's English stop list but carry no
# topical meaning in short news-style text. Kept deliberately small and
# domain-neutral so that no topic is "helped" by the list.
CUSTOM_STOPWORDS = {
    "said", "says", "new", "year", "today", "also", "make", "makes", "like",
    "use", "uses", "used", "using", "time", "long", "two", "three", "high",
    "higher", "good", "better", "keep", "told", "asked", "help", "helped",
    "half", "home", "every", "early", "late", "large", "faster", "longer",
    "expect", "expects", "expected", "reported", "started", "opened",
}
STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | CUSTOM_STOPWORDS

TOKEN_RE = re.compile(r"[a-z]{3,}")  # alphabetic tokens, length >= 3


def tokenize(text: str) -> list[str]:
    """Lowercase, keep alphabetic tokens of length >= 3, drop stopwords."""
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def build_plural_map(tokenised_docs: list[list[str]]) -> dict[str, str]:
    """Corpus-aware plural merging.

    A plural is merged into its singular ONLY if the singular form also
    occurs somewhere in the corpus, e.g. players -> player, companies ->
    company. Words like 'series' or 'diabetes' are left alone because
    'sery' / 'diabete' never occur. This avoids an NLTK/spaCy download
    while still preventing 'player' and 'players' from splitting one
    topic's probability mass across two vocabulary entries.
    """
    vocab = {t for doc in tokenised_docs for t in doc}
    mapping: dict[str, str] = {}
    for w in vocab:
        if w.endswith("ies") and w[:-3] + "y" in vocab:
            mapping[w] = w[:-3] + "y"
        elif (
            w.endswith("s")
            and not w.endswith(("ss", "us", "is"))
            and len(w) > 4
            and w[:-1] in vocab
        ):
            mapping[w] = w[:-1]
    return mapping


def preprocess(texts: list[str]) -> tuple[list[str], dict[str, str], list[list[str]]]:
    """Return (cleaned_texts, plural_map, tokens_per_doc)."""
    tokenised = [tokenize(t) for t in texts]
    plural_map = build_plural_map(tokenised)
    merged = [[plural_map.get(t, t) for t in doc] for doc in tokenised]
    cleaned = [" ".join(doc) for doc in merged]
    return cleaned, plural_map, merged


# ---------------------------------------------------------------------------
# 3. Document-term matrix
# ---------------------------------------------------------------------------


def build_dtm(cleaned_texts: list[str], min_df: int | None = None, max_df: float = 0.5):
    """Bag-of-words counts. LDA is a generative model of word COUNTS, so raw
    term frequencies are used, not TF-IDF.

    min_df: drop words appearing in fewer than min_df documents. Defaults to
            2 for corpora with >= 10 docs (a word seen in one document cannot
            tie documents together into a topic), else 1.
    max_df: drop words appearing in more than this fraction of documents
            (too common to separate topics).
    """
    n_docs = len(cleaned_texts)
    if min_df is None:
        min_df = 2 if n_docs >= 10 else 1
    if n_docs < 4:
        max_df = 1.0  # tiny inputs: keep everything
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b[a-z]{3,}\b", min_df=min_df, max_df=max_df)
    X = vectorizer.fit_transform(cleaned_texts)
    return X, vectorizer


# ---------------------------------------------------------------------------
# 4. LDA
# ---------------------------------------------------------------------------


def _fit_once(X, n_topics, seed, max_iter, doc_topic_prior, topic_word_prior):
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        max_iter=max_iter,
        evaluate_every=5,  # check perplexity every 5 iterations ...
        perp_tol=0.01,     # ... and stop once it stops improving
        random_state=seed,
        doc_topic_prior=doc_topic_prior,
        topic_word_prior=topic_word_prior,
    )
    return lda.fit(X)


def fit_lda(
    X,
    n_topics: int = 4,
    random_state: int = 42,
    max_iter: int = 300,
    n_restarts: int = 10,
    doc_topic_prior: float | None = None,
    topic_word_prior: float | None = None,
) -> LatentDirichletAllocation:
    """Fit scikit-learn LDA (variational Bayes, batch mode) with restarts.

    Why restarts: variational inference only finds a LOCAL optimum, and the
    starting point is random. On a 20-document corpus one unlucky start can
    merge two themes into one topic. So the model is fitted n_restarts
    times (seeds random_state, random_state+1, ...) and the run with the
    highest approximate log-likelihood, lda.score(X), is kept. This choice
    is fully unsupervised - it never looks at the true theme labels.

    Priors: None = scikit-learn's default of 1 / n_components for both
    alpha (doc_topic_prior) and eta (topic_word_prior).
    """
    best, best_score, best_seed = None, -np.inf, None
    for i in range(max(1, n_restarts)):
        seed = random_state + i
        lda = _fit_once(X, n_topics, seed, max_iter, doc_topic_prior, topic_word_prior)
        score = lda.score(X)
        if score > best_score:
            best, best_score, best_seed = lda, score, seed
    best.best_seed_ = best_seed            # which restart won
    best.best_loglik_ = float(best_score)  # its approximate log-likelihood
    return best


# ---------------------------------------------------------------------------
# 5. Top words per topic
# ---------------------------------------------------------------------------


def topic_word_distribution(lda: LatentDirichletAllocation) -> np.ndarray:
    """Row-normalise components_ to get P(word | topic)."""
    comp = lda.components_
    return comp / comp.sum(axis=1, keepdims=True)


def top_words(lda, feature_names, n: int = 5) -> pd.DataFrame:
    """Long-format table: topic, rank, word, weight (= P(word|topic))."""
    phi = topic_word_distribution(lda)
    rows = []
    for k, row in enumerate(phi):
        for rank, idx in enumerate(row.argsort()[::-1][:n], start=1):
            rows.append({"topic": k, "rank": rank, "word": feature_names[idx], "weight": float(row[idx])})
    return pd.DataFrame(rows)


def top_words_wide(top_df: pd.DataFrame) -> pd.DataFrame:
    """One row per topic: 'Topic 1 | w1, w2, w3, w4, w5'."""
    return (
        top_df.sort_values(["topic", "rank"])
        .groupby("topic")["word"]
        .apply(lambda s: ", ".join(s))
        .reset_index()
        .rename(columns={"word": "top_words"})
        .assign(topic=lambda d: d["topic"].map(lambda k: f"Topic {k + 1}"))
    )


# ---------------------------------------------------------------------------
# 6. Document-topic mixture
# ---------------------------------------------------------------------------


def doc_topic_table(lda, X, doc_ids) -> pd.DataFrame:
    theta = lda.transform(X)  # rows already sum to 1
    cols = [f"Topic {k + 1}" for k in range(theta.shape[1])]
    df = pd.DataFrame(theta, columns=cols)
    df.insert(0, "doc_id", list(doc_ids))
    df["dominant_topic"] = theta.argmax(axis=1) + 1
    df["dominant_prob"] = theta.max(axis=1)
    return df


# ---------------------------------------------------------------------------
# 7. Evaluation
# ---------------------------------------------------------------------------


def umass_coherence(X, top_df: pd.DataFrame, vectorizer) -> pd.DataFrame:
    """UMass topic coherence (Mimno et al., 2011), computed from the corpus
    itself: for each pair of top words (w_i ranked below w_j)

        score += log( (D(w_i, w_j) + 1) / D(w_j) )

    where D() counts documents containing the word(s). Values are <= 0;
    closer to 0 = top words co-occur more = more coherent topic.
    """
    B = (X > 0).astype(int).tocsc()
    vocab = vectorizer.vocabulary_
    out = []
    for k, grp in top_df.sort_values(["topic", "rank"]).groupby("topic"):
        ids = [vocab[w] for w in grp["word"]]
        score, pairs = 0.0, 0
        for i in range(1, len(ids)):
            for j in range(i):
                col_i = B[:, ids[i]].toarray().ravel()
                col_j = B[:, ids[j]].toarray().ravel()
                d_j = col_j.sum()
                d_ij = (col_i & col_j).sum()
                score += np.log((d_ij + 1) / d_j)
                pairs += 1
        out.append({"topic": f"Topic {k + 1}", "umass": score / max(pairs, 1)})
    return pd.DataFrame(out)


def label_topics(doc_topics: pd.DataFrame, true_themes) -> dict[int, str]:
    """Name each topic by the majority true theme of the documents for which
    it is dominant (only possible when labels are known)."""
    tmp = pd.DataFrame({"topic": doc_topics["dominant_topic"], "theme": list(true_themes)})
    labels = {}
    for k, grp in tmp.groupby("topic"):
        labels[int(k)] = grp["theme"].value_counts().idxmax()
    return labels


def theme_agreement(doc_topics: pd.DataFrame, true_themes) -> dict:
    """Compare discovered topics with the known themes the corpus was
    written around. LDA never sees these labels - they are only used here.

    purity: fraction of documents whose dominant topic's majority theme
            equals their own theme (1.0 = every topic is one theme)
    nmi / ari: label-permutation-invariant clustering agreement scores
    """
    pred = doc_topics["dominant_topic"].to_numpy()
    true = np.asarray(list(true_themes))
    contingency = pd.crosstab(pd.Series(pred, name="topic"), pd.Series(true, name="theme"))
    purity = contingency.max(axis=1).sum() / len(true)
    return {
        "purity": float(purity),
        "nmi": float(normalized_mutual_info_score(true, pred)),
        "ari": float(adjusted_rand_score(true, pred)),
        "contingency": contingency,
    }


def k_sweep(X, vectorizer, ks=range(2, 7), random_state: int = 42, true_themes=None, top_n: int = 5) -> pd.DataFrame:
    """Fit one LDA per K and report perplexity, mean UMass coherence and
    (if themes are known) purity. Note: perplexity is computed on the
    training documents because the corpus is too small to hold out data,
    so treat it as one signal alongside coherence and purity."""
    names = vectorizer.get_feature_names_out()
    rows = []
    for k in ks:
        lda = fit_lda(X, n_topics=k, random_state=random_state)
        tw = top_words(lda, names, n=top_n)
        row = {
            "k": k,
            "perplexity": float(lda.perplexity(X)),
            "mean_umass": float(umass_coherence(X, tw, vectorizer)["umass"].mean()),
        }
        if true_themes is not None:
            dt = doc_topic_table(lda, X, range(X.shape[0]))
            row["purity"] = theme_agreement(dt, true_themes)["purity"]
        rows.append(row)
    return pd.DataFrame(rows)


def seed_stability(X, n_topics: int, true_themes, seeds=range(10), n_restarts: int = 10) -> pd.DataFrame:
    """LDA is initialised randomly. For each seed, compare
    (a) a single LDA run   vs   (b) best-of-n_restarts (by log-likelihood),
    to show why restarts are needed on a small corpus."""
    rows = []
    for s in seeds:
        single = fit_lda(X, n_topics=n_topics, random_state=s, n_restarts=1)
        multi = fit_lda(X, n_topics=n_topics, random_state=1000 + 100 * s, n_restarts=n_restarts)
        p_single = theme_agreement(doc_topic_table(single, X, range(X.shape[0])), true_themes)["purity"]
        p_multi = theme_agreement(doc_topic_table(multi, X, range(X.shape[0])), true_themes)["purity"]
        rows.append({"seed": s, "purity_single_run": p_single, f"purity_best_of_{n_restarts}": p_multi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# One-call pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    texts: list[str],
    n_topics: int = 4,
    top_n: int = 5,
    random_state: int = 42,
    doc_ids=None,
    true_themes=None,
    min_df: int | None = None,
    max_df: float = 0.5,
    n_restarts: int = 10,
) -> dict:
    cleaned, plural_map, tokens = preprocess(texts)
    X, vectorizer = build_dtm(cleaned, min_df=min_df, max_df=max_df)
    if X.shape[1] < n_topics:
        raise ValueError(
            f"Only {X.shape[1]} vocabulary words survived filtering - add more "
            f"documents or lower min_df before asking for {n_topics} topics."
        )
    lda = fit_lda(X, n_topics=n_topics, random_state=random_state, n_restarts=n_restarts)
    names = vectorizer.get_feature_names_out()
    doc_ids = list(doc_ids) if doc_ids is not None else [f"D{i + 1:02d}" for i in range(len(texts))]

    top_df = top_words(lda, names, n=top_n)
    doc_topics = doc_topic_table(lda, X, doc_ids)
    coherence = umass_coherence(X, top_df, vectorizer)

    result = {
        "cleaned_texts": cleaned,
        "tokens": tokens,
        "plural_map": plural_map,
        "X": X,
        "vectorizer": vectorizer,
        "feature_names": names,
        "lda": lda,
        "top_words": top_df,
        "top_words_wide": top_words_wide(top_df),
        "doc_topics": doc_topics,
        "coherence": coherence,
        "perplexity": float(lda.perplexity(X)),
        "n_iter": int(lda.n_iter_),
        "best_seed": int(lda.best_seed_),
        "loglik": float(lda.best_loglik_),
        "labels": None,
        "agreement": None,
    }
    if true_themes is not None:
        result["labels"] = label_topics(doc_topics, true_themes)
        result["agreement"] = theme_agreement(doc_topics, true_themes)
        result["top_words_wide"]["likely_theme"] = [
            result["labels"].get(k + 1, "-") for k in range(n_topics)
        ]
    return result


# ---------------------------------------------------------------------------
# Saving outputs (CSV + PNG figures)
# ---------------------------------------------------------------------------


def save_outputs(
    result: dict,
    out_dir: str | Path = OUTPUT_DIR,
    sweep: pd.DataFrame | None = None,
    stability: pd.DataFrame | None = None,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []

    def _save_csv(df, name):
        p = out / name
        df.to_csv(p, index=False)
        saved.append(p)

    _save_csv(result["top_words_wide"], "topics_top5.csv")
    _save_csv(result["top_words"], "topic_word_weights.csv")
    _save_csv(result["doc_topics"].round(4), "doc_topic_distribution.csv")
    _save_csv(result["coherence"].round(4), "topic_coherence.csv")
    if sweep is not None:
        _save_csv(sweep.round(4), "model_selection.csv")
    if stability is not None:
        _save_csv(stability.round(4), "seed_stability.csv")

    # Figure 1: top-word bar chart per topic
    top = result["top_words"]
    k = top["topic"].nunique()
    fig, axes = plt.subplots(1, k, figsize=(3.4 * k, 3.2), sharex=False)
    axes = np.atleast_1d(axes)
    labels = result.get("labels") or {}
    for t, ax in zip(sorted(top["topic"].unique()), axes):
        d = top[top["topic"] == t].sort_values("weight")
        ax.barh(d["word"], d["weight"], color="#4C72B0")
        title = f"Topic {t + 1}"
        if labels.get(t + 1):
            title += f" ({labels[t + 1]})"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("P(word | topic)", fontsize=8)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    p = out / "topic_top_words.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    # Figure 2: document-topic heatmap
    dt = result["doc_topics"]
    topic_cols = [c for c in dt.columns if c.startswith("Topic ")]
    fig, ax = plt.subplots(figsize=(1.2 * len(topic_cols) + 2, 0.32 * len(dt) + 1.2))
    im = ax.imshow(dt[topic_cols].to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(topic_cols)), topic_cols, fontsize=8)
    ax.set_yticks(range(len(dt)), dt["doc_id"], fontsize=7)
    ax.set_title("Document-topic distribution (theta)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.05)
    fig.tight_layout()
    p = out / "doc_topic_heatmap.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    # Figure 3: model selection
    if sweep is not None:
        panels = [("perplexity", "Perplexity (lower = better)"), ("mean_umass", "Mean UMass coherence")]
        if "purity" in sweep:
            panels.append(("purity", "Purity vs known themes"))
        fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.2))
        best_k = int(sweep.loc[sweep["perplexity"].idxmin(), "k"])
        for ax, (col, title) in zip(np.atleast_1d(axes), panels):
            ax.plot(sweep["k"], sweep[col], marker="o", color="#4C72B0")
            ax.axvline(best_k, color="#DD8452", linestyle="--", linewidth=1)
            ax.set_xticks(sweep["k"])
            ax.set_xlabel("Number of topics (K)", fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.tick_params(labelsize=8)
        fig.suptitle(f"Choosing K (dashed line = lowest perplexity, K = {best_k})", fontsize=10)
        fig.tight_layout()
        p = out / "model_selection.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

    summary = {
        "n_documents": int(result["X"].shape[0]),
        "vocabulary_size": int(result["X"].shape[1]),
        "n_topics": int(result["lda"].n_components),
        "perplexity": round(result["perplexity"], 3),
        "em_iterations": result["n_iter"],
        "best_restart_seed": result["best_seed"],
        "approx_log_likelihood": round(result["loglik"], 3),
        "mean_umass": round(float(result["coherence"]["umass"].mean()), 4),
        "topics": {
            row["topic"]: row["top_words"] for _, row in result["top_words_wide"].iterrows()
        },
    }
    if result["agreement"]:
        summary.update(
            purity=round(result["agreement"]["purity"], 4),
            nmi=round(result["agreement"]["nmi"], 4),
            ari=round(result["agreement"]["ari"], 4),
            topic_labels={f"Topic {k}": v for k, v in result["labels"].items()},
        )
    p = out / "summary.json"
    p.write_text(json.dumps(summary, indent=2))
    saved.append(p)
    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="LDA topic modeling - top words per topic")
    ap.add_argument("--data", default=str(SAMPLE_CORPUS), help="CSV with a 'text' column, or TXT with one doc per line")
    ap.add_argument("--k", type=int, default=4, help="number of topics")
    ap.add_argument("--top-n", type=int, default=5, help="words to show per topic")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUTPUT_DIR))
    args = ap.parse_args()

    df = load_corpus(args.data)
    themes = df["true_theme"].tolist() if "true_theme" in df.columns else None
    res = run_pipeline(
        df["text"].tolist(), n_topics=args.k, top_n=args.top_n,
        random_state=args.seed, doc_ids=df["doc_id"], true_themes=themes,
    )
    sweep = k_sweep(res["X"], res["vectorizer"], true_themes=themes, random_state=args.seed, top_n=args.top_n)
    stability = seed_stability(res["X"], args.k, themes) if themes else None

    print(f"\nDocuments: {res['X'].shape[0]}  |  Vocabulary: {res['X'].shape[1]}  |  Topics: {args.k}")
    print(f"Perplexity: {res['perplexity']:.2f}  |  EM iterations: {res['n_iter']}  |  best restart seed: {res['best_seed']}\n")
    print(f"Top {args.top_n} words per topic")
    print("-" * 60)
    for _, row in res["top_words_wide"].iterrows():
        extra = f"   [{row['likely_theme']}]" if "likely_theme" in row else ""
        print(f"{row['topic']:<9}| {row['top_words']}{extra}")
    print("-" * 60)
    if res["agreement"]:
        a = res["agreement"]
        print(f"Purity {a['purity']:.3f} | NMI {a['nmi']:.3f} | ARI {a['ari']:.3f}")
    print(f"Mean UMass coherence: {res['coherence']['umass'].mean():.3f}")

    if stability is not None:
        cols = stability.columns[1:]
        print("Seed stability (mean purity over 10 seeds): "
              + ", ".join(f"{c} = {stability[c].mean():.2f}" for c in cols))

    saved = save_outputs(res, args.out, sweep=sweep, stability=stability)
    print(f"\nSaved {len(saved)} files to {Path(args.out).resolve()}")
    for p in saved:
        print("  -", p.name)


if __name__ == "__main__":
    main()
