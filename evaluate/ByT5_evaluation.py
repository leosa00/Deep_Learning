import pandas as pd
import ast
from collections import Counter
import sacrebleu

# load data
df_ref = pd.read_csv('./data/byt5_translated.csv')
df_hyp = pd.read_csv('./data/byt5_corrupt_translated.csv')


def extract(txt):
    try:
        lst = ast.literal_eval(txt)
        if isinstance(lst, list):
            return " ".join(lst)
    except:
        pass
    return txt


df_ref['es'] = df_ref['es'].apply(extract)

for col in ["es_corruption_1", "es_corruption_2", "es_corruption_3"]:
    df_hyp[col] = df_hyp[col].apply(extract)

# CHRF implementation


def character_ngrams(s, n):
    s = s.replace(" ", "")
    return [s[i:i+n] for i in range(len(s)-n+1)]


def chrf_score(ref, hyp, max_n=6, beta=1):
    total_prec, total_rec = 0.0, 0.0
    eps = 1e-16
    for n in range(1, max_n+1):
        ref_ngr = character_ngrams(ref, n)
        hyp_ngr = character_ngrams(hyp, n)
        ref_cnt = Counter(ref_ngr)
        hyp_cnt = Counter(hyp_ngr)
        overlap = sum(min(hyp_cnt[g], ref_cnt[g]) for g in hyp_cnt)
        prec = overlap / (len(hyp_ngr) + eps)
        rec = overlap / (len(ref_ngr) + eps)
        total_prec += prec
        total_rec += rec

    CHRP = total_prec / max_n
    CHRR = total_rec / max_n
    beta2 = beta * beta
    return (1 + beta2) * CHRP * CHRR / (beta2 * CHRP + CHRR + eps)


# Compute BLEU and CHRF scores
refs = df_ref['es'].tolist()

# Clean (self-BLEU)
bleu_clean = sacrebleu.corpus_bleu(refs, [refs]).score
chrf_clean = sum(chrf_score(r, r) for r in refs) / len(refs)
results = {"clean": {"BLEU": bleu_clean, "CHRF": 100 * chrf_clean}}

# Corruption levels
for i in [1, 2, 3]:
    hyps = df_hyp[f'es_corruption_{i}'].tolist()

    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf_vals = [chrf_score(ref, hyp) for ref, hyp in zip(refs, hyps)]
    results[f'corruption_{i}'] = {
        "BLEU": bleu,
        "CHRF": 100 * sum(chrf_vals) / len(chrf_vals)
    }

print("= BLEU & CHRF Scores =\n")
print(f"{'Dataset':<15}{'BLEU':<12}{'CHRF'}")
print("-" * 35)

for key, vals in results.items():
    print(f"{key:<15}{vals['BLEU']:<12.4f}{vals['CHRF']:.4f}")
