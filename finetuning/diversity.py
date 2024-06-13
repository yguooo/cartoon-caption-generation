import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

def compute_diversity(lst):
    # Compute the average cosine similarity between all pairs of sentences in the list
    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(lst)
    sim = np.inner(embeddings, embeddings)
    return np.mean(sim)

def compute_overall_diversity(filenanme):
    
    df = pd.read_csv(filenanme)
    df_train = df[df["info"] =="train"]
    df_test = df[df["info"] =="test"]

    train_diversity = []
    for _, row in df_train.iterrows():
        train_diversity.append(compute_diversity(list(row)[3:8]))
    test_diversity = []
    for _, row in df_test.iterrows():
        test_diversity.append(compute_diversity(list(row)[3:8]))
    train_diversity = 1 - np.mean(train_diversity)
    test_diversity = 1 - np.mean(test_diversity)
    print("train diversity: ", train_diversity, "test diversity: ", test_diversity)
    return train_diversity, test_diversity  

if __name__ == "__main__":
    ckpt_id = 1000
    print(ckpt_id)
    compute_overall_diversity('dpo_results_ckpt{}.csv'.format(str(ckpt_id)))