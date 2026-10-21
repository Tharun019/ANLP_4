# ANLP_4 – Topic Modeling with Latent Dirichlet Allocation (LDA)

**Course:** Advanced NLP (AML23702) · **Lab Exercise 4**

**Problem statement:** Apply Latent Dirichlet Allocation (LDA) to a small collection of text documents and display the top five words associated with each discovered topic.

## Result (sample corpus, K = 4)

| Topic | Top 5 words | Matches theme |
|---|---|---|
| Topic 1 | model, software, data, user, trained | Technology |
| Topic 2 | patient, doctor, hospital, heart, diet | Health |
| Topic 3 | team, player, match, coach, fans | Sports |
| Topic 4 | market, investors, company, stock, share | Finance |

Every one of the 20 documents was assigned to the topic matching the theme it was written about (purity = 1.00, NMI = 1.00, ARI = 1.00). LDA never saw those theme labels; they were used only to check the result afterwards.

## How it works

```
raw documents
   │  lowercase · alphabetic tokens (3+ letters) · stopword removal · plural merging
   ▼
cleaned text
   │  CountVectorizer (raw counts, min_df = 2, max_df = 0.5)
   ▼
document-term matrix  (20 docs × 71 words)
   │  LatentDirichletAllocation (batch variational Bayes, K = 4, 10 restarts → best log-likelihood)
   ▼
lda.components_  →  normalise rows → P(word | topic) → top 5 words per topic
lda.transform(X) →  P(topic | document) → dominant topic per document
   │
   ▼
evaluation: perplexity · UMass coherence · purity / NMI / ARI · K sweep · seed stability
```

Why each choice was made:

- **Raw counts, not TF-IDF.** LDA is a generative model of word counts, so it expects a bag of words.
- **Restarts.** Variational inference only reaches a local optimum. On 20 documents, a single random start sometimes merges two themes (mean purity 0.81 over 10 seeds). Keeping the best of 10 restarts by log-likelihood gave purity 1.00 for every seed tested, and it never uses the labels.
- **Choosing K.** Perplexity is lowest at K = 4, the number of themes the corpus was written around.

## Project structure

```
ANLP_4/
├── pipeline.py           # all NLP + LDA logic, also a CLI
├── app.py                # Streamlit "Topic Explorer" web app
├── build_notebook.py     # builds and executes notebook.ipynb
├── notebook.ipynb        # step-by-step run with real outputs
├── sample_data/
│   └── corpus.csv        # 20 synthetic documents (doc_id, true_theme, text)
├── outputs/              # CSVs, PNG figures and summary.json written by the pipeline
├── requirements.txt
└── README.md
```

## Run it

```bash
pip install -r requirements.txt
python pipeline.py              # prints top-5 words per topic, writes outputs/
python pipeline.py --k 3        # try a different number of topics
python build_notebook.py        # regenerate the executed notebook
streamlit run app.py            # interactive app
```

Your own documents: `python pipeline.py --data my_docs.txt` (one document per line), or a CSV file with a `text` column. In the app, use **Paste text** or **Upload file**.

## Tools

Python 3 · scikit-learn (CountVectorizer, LatentDirichletAllocation, metrics) · pandas · NumPy · Matplotlib · Plotly · Streamlit · nbformat / nbconvert

## References

- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. *JMLR*, 3, 993–1022. https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
- Hoffman, M., Bach, F., & Blei, D. (2010). Online Learning for Latent Dirichlet Allocation. *NeurIPS*.
- Mimno, D., Wallach, H., Talley, E., Leenders, M., & McCallum, A. (2011). Optimizing Semantic Coherence in Topic Models. *EMNLP*.
- scikit-learn – LatentDirichletAllocation: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
- scikit-learn User Guide – LDA: https://scikit-learn.org/stable/modules/decomposition.html#latentdirichletallocation
- Streamlit documentation: https://docs.streamlit.io
