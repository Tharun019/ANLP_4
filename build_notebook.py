"""
Builds and EXECUTES notebook.ipynb, so the notebook in the repo always
contains real outputs from pipeline.py.

Run:  python build_notebook.py
"""

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parent
md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

cells = [
    md("# ANLP_4 - Topic Modeling with Latent Dirichlet Allocation\n"
       "**Problem statement:** Apply LDA to a small collection of text documents and display the "
       "top five words associated with each discovered topic.\n\n"
       "This notebook walks through every stage of `pipeline.py` and records its real outputs."),
    code("import pandas as pd\n"
         "import pipeline as P\n"
         "pd.set_option('display.max_colwidth', 120)"),

    md("## 1. Load the corpus\n"
       "20 short synthetic news-style documents, 5 each about Sports, Finance, Health and Technology. "
       "The `true_theme` column is **never given to LDA** - it is only used afterwards to check the topics."),
    code("df = P.load_corpus(P.SAMPLE_CORPUS)\n"
         "print(df['true_theme'].value_counts().to_dict())\n"
         "df.head(6)"),

    md("## 2. Preprocessing\n"
       "Lowercase -> keep alphabetic tokens of 3+ letters -> remove stopwords (scikit-learn list + a small "
       "custom list) -> merge plurals into singulars when the singular also occurs in the corpus."),
    code("cleaned, plural_map, tokens = P.preprocess(df['text'].tolist())\n"
         "print('RAW    :', df['text'][0])\n"
         "print('CLEANED:', cleaned[0])\n"
         "print('\\nPlural merges:', dict(sorted(plural_map.items())))"),

    md("## 3. Document-term matrix (bag of words)\n"
       "LDA models word **counts**, so `CountVectorizer` is used (not TF-IDF). "
       "`min_df=2` drops words seen in only one document; `max_df=0.5` drops words in more than half of them."),
    code("X, vectorizer = P.build_dtm(cleaned)\n"
         "names = vectorizer.get_feature_names_out()\n"
         "print('DTM shape (documents x words):', X.shape)\n"
         "freq = pd.Series(X.sum(axis=0).A1, index=names).sort_values(ascending=False)\n"
         "freq.head(15)"),

    md("## 4. Fit LDA\n"
       "scikit-learn `LatentDirichletAllocation`, batch variational Bayes, K = 4. "
       "Fitted 10 times with different seeds; the run with the highest approximate log-likelihood "
       "(`lda.score(X)`) is kept - an unsupervised choice that does not look at the labels."),
    code("lda = P.fit_lda(X, n_topics=4, random_state=42, n_restarts=10)\n"
         "print('Winning seed      :', lda.best_seed_)\n"
         "print('Log-likelihood    :', round(lda.best_loglik_, 2))\n"
         "print('EM iterations     :', lda.n_iter_)\n"
         "print('Perplexity        :', round(lda.perplexity(X), 2))\n"
         "print('components_ shape :', lda.components_.shape, '(topics x words)')"),

    md("## 5. Top five words per topic  *(main deliverable)*\n"
       "Each row of `lda.components_` is normalised to P(word | topic); the 5 largest values are the top words."),
    code("top = P.top_words(lda, names, n=5)\n"
         "for t, grp in top.groupby('topic'):\n"
         "    words = ', '.join(f\"{w} ({p:.3f})\" for w, p in zip(grp['word'], grp['weight']))\n"
         "    print(f'Topic {t + 1}: {words}')"),
    code("P.top_words_wide(top)"),

    md("## 6. Document-topic mixture\n"
       "`lda.transform(X)` gives P(topic | document). The dominant topic is the largest entry."),
    code("dt = P.doc_topic_table(lda, X, df['doc_id'])\n"
         "dt.assign(true_theme=df['true_theme']).round(3)"),

    md("## 7. Evaluation\n"
       "### 7.1 UMass coherence (from the corpus itself, higher / closer to 0 = better)"),
    code("P.umass_coherence(X, top, vectorizer).round(3)"),
    md("### 7.2 Agreement with the known themes"),
    code("agr = P.theme_agreement(dt, df['true_theme'])\n"
         "labels = P.label_topics(dt, df['true_theme'])\n"
         "print({f'Topic {k}': v for k, v in labels.items()})\n"
         "print(f\"Purity {agr['purity']:.3f} | NMI {agr['nmi']:.3f} | ARI {agr['ari']:.3f}\")\n"
         "agr['contingency']"),
    md("### 7.3 Choosing K\n"
       "Perplexity here is measured on the training documents (the corpus is too small to hold data out)."),
    code("sweep = P.k_sweep(X, vectorizer, ks=range(2, 7), true_themes=df['true_theme'])\n"
         "sweep.round(3)"),
    md("### 7.4 Why restarts? Seed stability"),
    code("stab = P.seed_stability(X, 4, df['true_theme'])\n"
         "print(stab.iloc[:, 1:].mean().round(3).to_dict())\n"
         "stab"),

    md("## 8. Save outputs (CSV + figures) to `outputs/`"),
    code("res = P.run_pipeline(df['text'].tolist(), n_topics=4, doc_ids=df['doc_id'], true_themes=df['true_theme'])\n"
         "saved = P.save_outputs(res, sweep=sweep, stability=stab)\n"
         "[p.name for p in saved]"),
    code("from IPython.display import Image, display\n"
         "for f in ['topic_top_words.png', 'doc_topic_heatmap.png', 'model_selection.png']:\n"
         "    display(Image(filename=str(P.OUTPUT_DIR / f)))"),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}

print("Executing notebook ... (about 30-60 s)")
ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(ROOT)}})
out = ROOT / "notebook.ipynb"
nbformat.write(nb, out)
print(f"Saved executed notebook -> {out}")
