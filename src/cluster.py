from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import Config


def load_embeddings(fold_num, input_folder="results/bert_embeddings_reduced"):
    input_path = f"{input_folder}/bert_embeddings_reduced_fold{fold_num}.csv"
    df = pd.read_csv(input_path)

    df["reduced_embeddings"] = df["reduced_embeddings"].apply(eval)

    embeddings = pd.DataFrame(df["reduced_embeddings"].tolist(), index=df.index)

    return df, embeddings


def apply_kmeans(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    return clusters, kmeans


def extract_sentence(row, bert_input_df):
    idx = row["idx"]

    matched_sentences = bert_input_df.loc[
        bert_input_df["idx"] == idx,
        "sentence",
    ]

    assert len(matched_sentences) == 1, f"Found multiple sentences for idx: {idx}"

    return matched_sentences.values[0]


def sample_sentences_from_clusters(
    df, bert_input_df, n_clusters=5, samples_per_cluster=8
):
    sampled_sentences = []

    for cluster_num in range(n_clusters):
        cluster_df = df[df["cluster"] == cluster_num]
        cluster_size = len(cluster_df)
        
        if cluster_size < samples_per_cluster:
            sampled_df = cluster_df
            print(f"Cluster {cluster_num} has only {cluster_size} samples, using all of them.")
        else:
            sampled_df = cluster_df.sample(n=samples_per_cluster, random_state=42)

        sampled_df.loc[:, "sentence"] = sampled_df.apply(
            lambda row: extract_sentence(row, bert_input_df), axis=1
        )

        sampled_sentences.extend(
            list(
                zip(
                    sampled_df["idx"],
                    sampled_df["sentence"],
                    [cluster_num] * samples_per_cluster,
                )
            )
        )

    return sampled_sentences


def find_best_k_silhouette(embeddings, min_clusters=2, max_clusters=10):
    silhouette_scores = []
    k_values = list(range(min_clusters, max_clusters + 1))

    best_k = min_clusters
    best_silhouette_score = -1

    for k in k_values:
        clusters, kmeans = apply_kmeans(embeddings, n_clusters=k)
        
        unique_labels = len(set(clusters))
        if unique_labels == 1:
            print(f"Warning: Only one cluster found for k={k}, skipping silhouette score.")
            silhouette_scores.append(None)
            continue
        
        score = silhouette_score(embeddings, clusters)
        silhouette_scores.append(score)

        if score > best_silhouette_score:
            best_k = k
            best_silhouette_score = score

    return k_values, silhouette_scores, best_k, best_silhouette_score


def plot_silhouette_scores_for_all_folds(
    silhouette_data,
    output_path="results/kmeans_results/silhouette_scores_all_folds.png",
):
    plt.figure(figsize=(10, 6))

    for fold_num, (k_values, scores) in enumerate(silhouette_data, start=1):
        plt.plot(k_values, scores, marker="o", label=f"Fold {fold_num}")

    plt.title("Silhouette Score vs Number of Clusters (for all folds)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)
    print(f"Silhouette score plot saved to {output_path}")
    plt.clf()  # Clear the figure for future plots


def process_kmeans_for_fold(
    fold_num,
    min_clusters=2,
    max_clusters=10,
    samples_per_cluster=10,
    input_folder="results/bert_embeddings_reduced",
    output_folder="results/kmeans_results",
    bert_inputs_folder="results/bert_inputs",
):
    df, embeddings = load_embeddings(fold_num, input_folder)

    # Find best k based on silhouette score
    k_values, silhouette_scores, best_k, best_silhouette_score = find_best_k_silhouette(
        embeddings, min_clusters=min_clusters, max_clusters=max_clusters
    )

    clusters, kmeans_model = apply_kmeans(embeddings, n_clusters=best_k)

    df["cluster"] = clusters

    bert_input_path = f"{bert_inputs_folder}/bert_input_fold{fold_num}.csv"
    bert_input_df = pd.read_csv(bert_input_path)

    sampled_sentences = sample_sentences_from_clusters(
        df, bert_input_df, n_clusters=best_k, samples_per_cluster=samples_per_cluster
    )

    sampled_df = pd.DataFrame(sampled_sentences, columns=["idx", "sentence", "cluster"])

    output_path = f"{output_folder}/kmeans_sampled_sentences_fold{fold_num}.csv"
    sampled_df.to_csv(output_path, index=False)

    print(
        f"Processed KMeans for fold {fold_num} with k={best_k}, sampled {best_k * samples_per_cluster} sentences, and saved results to {output_path}"
    )

    return k_values, silhouette_scores


if __name__ == "__main__":
    min_clusters = 2
    max_clusters = 10
    num_folds = Config.num_folds

    silhouette_data = []

    for i in range(1, num_folds + 1):
        k_values, silhouette_scores = process_kmeans_for_fold(i, min_clusters=min_clusters, max_clusters=max_clusters)
        silhouette_data.append((k_values, silhouette_scores))

    plot_silhouette_scores_for_all_folds(silhouette_data)
