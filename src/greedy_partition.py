import pandas as pd
import itertools

def greedy_partition(arr, num_subsets=5):
    """
    Perform a greedy partitioning of the input array into `num_subsets` subsets.
    
    Parameters:
    - arr: A list of integers representing the counts to be partitioned.
    - num_subsets: The number of subsets to partition the array into (default is 5).
    
    Returns:
    - subsets: A list of lists, where each inner list represents a subset.
    - subset_sums: A list of integers representing the sum of each subset.
    - subset_indices: A list of lists, where each inner list contains the indices of the elements in the original array.
    - imbalance: An integer representing the approximate imbalance among the subsets.
    """
    arr.sort(reverse=True)  # Sort the array in descending order
    subsets = [[] for _ in range(num_subsets)]
    subset_sums = [0] * num_subsets
    subset_indices = [[] for _ in range(num_subsets)]

    for i, num in enumerate(arr):
        min_idx = subset_sums.index(min(subset_sums))  # Find the subset with the minimum sum
        subsets[min_idx].append(num)
        subset_sums[min_idx] += num
        subset_indices[min_idx].append(i)

    # Calculate imbalance as the sum of absolute differences between all pairs of subset sums
    imbalance = sum(abs(x - y) for x, y in itertools.combinations(subset_sums, 2))

    return subsets, subset_sums, subset_indices, imbalance

def save_partition_to_csv(essay_ids, subset_indices, subset_sums, filename="results/partition_results.csv"):
    """
    Save partition results to a CSV file.
    
    Parameters:
    - essay_ids: A list of essay IDs corresponding to the counts.
    - subset_indices: A list of lists, where each inner list contains the indices of essays in each subset.
    - subset_sums: A list of integers representing the sum of each subset.
    - filename: The filename to save the results as a CSV file (default is 'partition_results.csv').
    """
    data = {
        "subset": [],
        "essay_id": [],
        "subset_sum": []
    }
    for i, (indices, subset_sum) in enumerate(zip(subset_indices, subset_sums)):
        for idx in indices:
            data["subset"].append(f"Subset {i+1}")
            data["essay_id"].append(essay_ids[idx])
            data["subset_sum"].append(subset_sum)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Partition results saved to {filename}")


def main(file_path='datasets/sentences_labels_all_0906_essay_matters.csv', save_csv=True):
    """
    Main function to execute the greedy partitioning and save the results.
    
    Parameters:
    - file_path: Path to the input CSV file (default is 'datasets/sentences_labels_all_0906_essay_matters.csv').
    - save_csv: Boolean to indicate if the results should be saved to a CSV file (default is True).
    - save_json: Boolean to indicate if the results should be saved to a JSON file (default is True).
    """
    # Load dataset
    df = pd.read_csv(file_path)

    # Calculate label distribution (essay counts)
    label_distribution = df['essay_id'].value_counts()
    label_dict = label_distribution.to_dict()
    essay_ids = list(label_dict.keys())
    counts = list(label_dict.values())

    # Perform greedy partitioning
    subsets, subset_sums, subset_indices, imbalance = greedy_partition(counts)

    # Print results
    print(f"Subset sums: {subset_sums}")
    print(f"Approximate minimum imbalance: {imbalance}")
    print("-" * 50)

    # Save results to files
    if save_csv:
        save_partition_to_csv(essay_ids, subset_indices, subset_sums)

# Optional testing framework for verifying functionality
def test_greedy_partition():
    """
    Test cases to verify the correctness of the greedy partitioning algorithm.
    """
    test_cases = [
        [2, 4, 5, 6, 8, 10, 12, 14, 16, 18],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [5, 9, 7, 3, 2, 6, 8, 4, 1, 10],
        [15, 5, 10, 20, 25, 30],
        [3, 7, 2, 5, 8, 6, 4, 9, 1, 10],
        [1, 2, 3, 4, 5],
        [0, 0, 0, 0, 0],
        [2, 2, 2, 2, 2]
    ]
    
    for idx, case in enumerate(test_cases):
        subsets, subset_sums, subset_indices, imbalance = greedy_partition(case)
        print(f"Test case {idx + 1}: Array = {case}")
        print(f"Subset sums: {subset_sums}")
        print(f"Subset indices: {subset_indices}")
        print(f"Approximate minimum imbalance: {imbalance}")
        print("-" * 50)

if __name__ == '__main__':
    main()
