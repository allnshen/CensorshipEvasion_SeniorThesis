================================================================================
  Contextual Bandits Under Censorship
  Allen Shen — Thesis Codebase
================================================================================

OVERVIEW
--------
This repository contains the code for a thesis investigating whether a
contextual bandit agent can learn to automatically evade online content
moderation. The system uses a LinUCB bandit with a frozen neural backbone to
select among 17 text-rewriting strategies ("arms") that attempt to transform
censored social media posts so they pass an automated content filter while
preserving their original meaning.

The pipeline has four stages, each corresponding to a notebook:

  1. Data preprocessing and translation of Chinese social media posts.
  2. Batch translation at scale via the Kaggle platform.
  3. Training and evaluation of the contextual bandit agent.
  4. Generation of figures and statistical analysis for the thesis.


DATA SOURCE
-----------
The dataset comes from the Weiboscope project, which archives censored and
uncensored posts from Sina Weibo (a major Chinese social media platform). Posts
flagged as censored by Weiboscope are used as the training signal; posts not
flagged are treated as uncensored baselines.


NOTEBOOK DESCRIPTIONS
---------------------

1. data_preprocessing.ipynb
   - Reads raw Weiboscope CSV files from Google Drive.
   - Separates censored and uncensored posts.
   - Cleans text (removes URLs, special characters, whitespace artifacts).
   - Translates Chinese text to English using the googletrans library.
   - Saves cleaned and translated text as pickle files.
   - Includes an exploratory data visualization cell.

2. kaggle_translation.ipynb
   - Performs large-scale Chinese-to-English translation on the Kaggle
     platform using GPT-4.1-nano via the OpenAI async API.
   - Implements retry logic with exponential backoff for rate limits.
   - Processes texts in concurrent batches (semaphore-controlled).
   - Outputs a CSV mapping original Chinese text to English translations.

3. revised_main_loop.ipynb  (core experiment)
   - Loads the translated English dataset and top-200 keyword features.
   - Builds context vectors by concatenating SBERT embeddings
     (all-MiniLM-L6-v2, 384-d) with binary keyword-presence features
     (200-d), producing a 585-dimensional feature vector per post.
   - Defines the EvasionActionSpace with 17 arms:
       Arms 0-14  — Semantic rewrites via GPT-4o-mini (synonym swap,
                     hypernym/hyponym replacement, reordering, paraphrase,
                     acronym substitution, filler insertion, voice flip,
                     informal/TikTok slang, academic tone, fragmentation,
                     phonetic spelling, code words/metaphors, negated
                     opposites, leetspeak rephrasing).
       Arm  15    — Distraction augmentation (prepends benign filler
                     sentences; rule-based).
       Arm  16    — Homoglyph substitution (replaces Latin characters with
                     visually similar Unicode glyphs; rule-based).
   - Oracle: Alibaba Cloud Content Moderation API. Returns a moderation
     label for each text; "pass" = evasion success.
   - Reward function combines evasion success, cosine similarity to the
     original text, and an action cost penalty.
   - The Backbone is a small feedforward network (585 -> 256 -> 64) pre-
     trained as a binary censored/uncensored classifier, then frozen.
   - LinUCB operates on the 64-d latent features from the frozen backbone,
     maintaining per-arm A and b matrices for upper confidence bound
     selection.
   - Training runs for 5 epochs over a pre-filtered candidate set of up to
     20,000 texts, with validation every 100 steps on a held-out set of 100
     texts.
   - Logs training history and validation curves to CSV; saves model
     checkpoints as pickle files.
   - Includes a holdout evaluation cell that exhaustively tests every arm on
     each validation text.

4. results_and_analysis.ipynb
   - Loads training logs and validation curves from CSV.
   - Generates publication-quality figures for Sections 6 and 7 of the
     thesis, including rolling evasion rate, validation curves, arm
     selection distributions, and label analysis.
   - Uses matplotlib and seaborn with 300 DPI export settings.


RUNTIME ENVIRONMENT
-------------------
All notebooks are designed to run in Google Colab with Google Drive mounted at
/content/drive/MyDrive/thesis/. The Kaggle translation notebook runs on the
Kaggle platform instead.

Required Python packages:
  - openai                (GPT-4o-mini rewrites and translation)
  - sentence-transformers (SBERT embeddings)
  - torch                 (backbone network)
  - pandas, numpy         (data handling)
  - matplotlib, seaborn   (visualization)
  - googletrans           (initial small-scale translation)
  - alibabacloud-green    (Alibaba content moderation oracle)
  - oss2                  (Alibaba Cloud SDK dependency)
  - python-Levenshtein    (string distance)
  - tqdm                  (progress bars)

API keys required (stored as Colab/Kaggle secrets):
  - OPENAI_API_KEY          — OpenAI API access
  - ALIBABA_KEY_ID          — Alibaba Cloud access key ID
  - ALIBABA_KEY_SECRET      — Alibaba Cloud access key secret


FILE STRUCTURE
--------------
Codebase/
  data_preprocessing.ipynb    — Data cleaning and initial translation
  kaggle_translation.ipynb    — Large-scale translation on Kaggle
  revised_main_loop.ipynb     — Bandit training and evaluation
  results_and_analysis.ipynb  — Thesis figures and analysis


REPRODUCIBILITY NOTES
---------------------
- The pre-filtering step caches oracle labels to avoid redundant API calls;
  cached results are loaded automatically on re-run.
- Validation uses temperature=0.0 for deterministic LLM outputs and a fixed
  random seed (999) for reproducible rule-based actions.
- Model checkpoints and training logs are saved incrementally so training can
  be resumed after interruption.
