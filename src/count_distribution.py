import pandas as pd

from config import Config


def calculate_acceptable_unacceptable_ratio(concatenated_df, fold_num):
    cu_columns = [f"CU{i}" for i in range(6)]

    all_unacceptable = (concatenated_df[cu_columns] == "unacceptable").all(axis=1)

    any_acceptable = (concatenated_df[cu_columns] == "acceptable").any(axis=1)

    num_all_unacceptable = all_unacceptable.sum()
    num_any_acceptable = any_acceptable.sum()

    total_rows = len(concatenated_df)

    unacceptable_ratio = num_all_unacceptable / total_rows
    acceptable_ratio = num_any_acceptable / total_rows

    print(f"Total rows: {total_rows}")
    print(f"All unacceptable: {num_all_unacceptable} ({unacceptable_ratio:.2%})")
    print(f"Any acceptable: {num_any_acceptable} ({acceptable_ratio:.2%})")

    ratio_output_path = (
        f"results/concatenated_results/ratio_results/ratio_fold{fold_num}.csv"
    )
    ratio_df = pd.DataFrame(
        {
            "total_rows": [total_rows],
            "all_unacceptable_count": [num_all_unacceptable],
            "all_unacceptable_ratio": [unacceptable_ratio],
            "any_acceptable_count": [num_any_acceptable],
            "any_acceptable_ratio": [acceptable_ratio],
        }
    )
    ratio_df.to_csv(ratio_output_path, index=False)

    return unacceptable_ratio, acceptable_ratio


def calculate_value_counts_for_cus(concatenated_df, fold_num):
    cu_columns = [f"CU{i}" for i in range(6)]

    value_counts_df = pd.DataFrame()

    for cu_col in cu_columns:
        value_counts = concatenated_df[cu_col].value_counts()
        value_counts_df[f"{cu_col}_count"] = value_counts

    for cu_col in cu_columns:
        value_counts = concatenated_df[cu_col].value_counts()
        
        percentages = value_counts / len(concatenated_df) * 100
        
        percentage_with_symbol = percentages.apply(lambda x: f"{x:.2f}%")

        value_counts_df[f"{cu_col}_percentage"] = percentage_with_symbol

    value_counts_output_path = f"results/concatenated_results/value_counts_results/value_counts_fold{fold_num}.csv"
    value_counts_df.to_csv(value_counts_output_path)

    print(f"Value counts for fold {fold_num} saved to {value_counts_output_path}")


if __name__ == "__main__":
    num_folds = Config.num_folds

    for fold_num in range(1, num_folds + 1):
        concatenated_df = pd.read_csv(
            f"results/concatenated_results/concatenated_fold{fold_num}.csv"
        )

        calculate_acceptable_unacceptable_ratio(concatenated_df, fold_num)

        calculate_value_counts_for_cus(concatenated_df, fold_num)
