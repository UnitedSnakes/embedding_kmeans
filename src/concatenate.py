import pandas as pd

from config import Config


def load_raw_folds(fold_num, input_folder="results/raw_folds"):
    raw_fold_path = f"{input_folder}/raw_fold{fold_num}.csv"
    raw_fold_df = pd.read_csv(raw_fold_path)
    return raw_fold_df


def load_kmeans_sentences(fold_num, input_folder="results/kmeans_results"):
    kmeans_path = f"{input_folder}/kmeans_sampled_sentences_fold{fold_num}.csv"
    kmeans_df = pd.read_csv(kmeans_path)
    kmeans_df['idx'] = kmeans_df['idx'].astype(float).astype(int)
    kmeans_df.drop(columns=["cluster"], inplace=True)
    return kmeans_df


def find_sentences_with_acceptable_cu(fold_num, input_folder="results/raw_folds"):
    raw_fold_df = load_raw_folds(fold_num, input_folder)

    # Select rows where any of the CU columns is "acceptable"
    cu_columns = [f"CU{i}" for i in range(6)]
    acceptable_df = raw_fold_df[(raw_fold_df[cu_columns] == "acceptable").any(axis=1)]

    return acceptable_df


def restore_missing_labels(kmeans_df, raw_fold_df):
    # Merge based on sentence matching
    merged_df = pd.merge(kmeans_df, raw_fold_df, on="sentence", how="left")

    # Find sentences where labels are missing (i.e., NaN in the label columns)
    missing_labels_df = merged_df[
        merged_df[["essay_id", "essay_type", "CU0", "CU1", "CU2", "CU3", "CU4", "CU5"]]
        .isnull()
        .any(axis=1)
    ]

    if not missing_labels_df.empty:
        print(f"Warning: {len(missing_labels_df)} sentences are missing labels.")
        print(
            missing_labels_df
        )  # Output missing sentences for debug purposes

    return merged_df


def concatenate_sentences(
    fold_num,
    kmeans_folder="results/kmeans_results",
    raw_folds_folder="results/raw_folds",
    output_folder="results/concatenated_results",
):
    kmeans_df = load_kmeans_sentences(fold_num, kmeans_folder)
    raw_fold_df = load_raw_folds(fold_num, raw_folds_folder)

    kmeans_with_labels = restore_missing_labels(kmeans_df, raw_fold_df)

    acceptable_df = find_sentences_with_acceptable_cu(fold_num, raw_folds_folder)

    # Concatenate the KMeans sentences with acceptable sentences
    concatenated_df = pd.concat([kmeans_with_labels, acceptable_df], ignore_index=True)

    output_path = f"{output_folder}/concatenated_fold{fold_num}.csv"
    concatenated_df.to_csv(output_path, index=False)

    print(f"Fold {fold_num} processed and saved to {output_path}")


if __name__ == "__main__":
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)

    num_folds = Config.num_folds

    for fold_num in range(1, num_folds + 1):
        concatenate_sentences(fold_num)
