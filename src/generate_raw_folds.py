import pandas as pd
import pandas.testing as pdt

from config import Config

if __name__ == "__main__":
    partition_df = pd.read_csv("results/partition_results.csv")
    partition_dict = dict(zip(partition_df["essay_id"], partition_df["subset"]))

    original_df = pd.read_csv("datasets/sentences_labels_all_0906_essay_matters.csv")

    folds_data = {f"fold{i + 1}": [] for i in range(Config.num_folds)}

    def assign_to_fold(row):
        essay_id = row["essay_id"]
        assert essay_id in partition_dict
        fold_number = int(partition_dict[essay_id].split(" ")[1])
        folds_data[f"fold{fold_number}"].append(row.copy())

    original_df.apply(assign_to_fold, axis=1)

    check_all_folds_df = pd.DataFrame()

    for i in range(1, Config.num_folds + 1):
        fold_df = pd.DataFrame(folds_data[f"fold{i}"])
        fold_df.to_csv(f"results/raw_folds/raw_fold{i}.csv", index=False)
        check_all_folds_df = pd.concat([check_all_folds_df, fold_df], ignore_index=True)

    try:
        sorted_original_df = original_df.sort_values(by=["essay_id", "essay_type"]).reset_index(
            drop=True
        )
        sorted_check_all_folds_df = check_all_folds_df.sort_values(
            by=["essay_id", "essay_type"]
        ).reset_index(drop=True)

        pdt.assert_frame_equal(sorted_check_all_folds_df, sorted_original_df)
        print(
            "Assertion passed: Combined DataFrame matches the original DataFrame (ignoring row order)."
        )
    except AssertionError as e:
        print(
            "Assertion failed: Combined DataFrame does not match the original DataFrame (even after sorting)."
        )
        print(e)

    print("All folds saved to disk.")
