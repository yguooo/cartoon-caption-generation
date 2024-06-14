import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

def lines_to_ngrams(lines, n=3):
    ngrams = []
    for s in lines:
        s = str(s)
        words = [e for e in s.replace('.','').replace('\n','').split(' ') if e != '']
        ngrams.append([tuple(words[i:i + n]) for i in range(len(words) - n + 1)])
    return ngrams

VOCAB_SIZE = 32000 # The Vocab size of Mistral 7B

def compute_ead(ngrams): 
    # Compute the Expectation-Adjusted Distinct (EAD) of a list of n-grams
    ngrams = [item for sublist in ngrams for item in sublist] # Flatten the list 
    N = len(set(ngrams))
    C = len(ngrams)
    V = VOCAB_SIZE

    try:
        ead = N / (V * (1 - ((V - 1) / V) ** C))
    except ZeroDivisionError:
        ead = 0.0
    return ead

def compute_averageEAD(df, start_col=2, end_col=12, n_min=1, n_max=5): 
    diversity = []
    for _, row in df.iterrows():
        ead = []
        for n in range(n_min, n_max+1):
            lines = list(row)[start_col:end_col] 
            ngrams = lines_to_ngrams(lines, n)
            ead.append(compute_ead(ngrams))
        diversity.append(np.mean(ead))
    diversity = np.mean(diversity)
    print("Averaged EAD: ", diversity)
    return diversity


def compute_diversity(lst):
    # Compute the average cosine similarity between all pairs of sentences in the list
    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(lst)
    sim = np.inner(embeddings, embeddings)
    return np.mean(sim)

def compute_overall_diversity(df, start_col=2, end_col=12):    
    diversity = []

    for _, row in df.iterrows():
        diversity.append(compute_diversity(list(row)[start_col:end_col]))
    diversity = 1 - np.mean(diversity)
    print("diversity: ", diversity)
    return diversity

if __name__ == "__main__":

    files = [
            # "your/generation/file.csv"
            "/data/yguo/myoutput/generation/sft_gen10.csv"
        ]
    # Use start_col = 2 for non-llava dataset and start_col = 3 for llava dataset, since llava requires an additional 
    # column for the image info.
    
    for file in files:
        print(file)
        df = pd.read_csv(file)
        compute_overall_diversity(df)
        compute_averageEAD(df)

