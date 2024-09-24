import pandas as pd

from config import Config


def process_folds_for_bert(
    fold_num,
    input_folder="results/raw_folds",
    output_folder="results/bert_inputs",
    frac=1,
):
    fold_df = pd.read_csv(f"{input_folder}/raw_fold{fold_num}.csv")

    cu_columns = [f"CU{i}" for i in range(6)]
    unacceptable_df = fold_df[(fold_df[cu_columns] == "unacceptable").all(axis=1)]

    sample_df = unacceptable_df.sample(frac=frac, random_state=42)

    selected_df = sample_df.reset_index(drop=False)[["index", "sentence"]]
    selected_df.rename(columns={"index": "idx"}, inplace=True)

    output_path = f"{output_folder}/bert_input_fold{fold_num}.csv"
    selected_df.reset_index(drop=True).to_csv(output_path, index=False)

    total_len = len(fold_df)
    sample_len = len(sample_df)
    print(
        f"Fold {fold_num} processed: {len(sample_df)} ({frac:.2%}) fully unacceptable samples saved to {output_path}.",
        f"Acceptable rate: {1 - sample_len / total_len:.2%}.",
        f"Unacceptable rate: {sample_len / total_len:.2%}.",
    )


if __name__ == "__main__":
    partition_df = pd.read_csv("results/partition_results.csv")

    for i in range(1, Config.num_folds + 1):
        process_folds_for_bert(i)
