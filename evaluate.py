import json

def calculate_manual_metrics(data_path):
    """
    Manually calculate micro F1 score, precision, and recall for multiple classes.

    Parameters:
    - data: A list of dictionaries, each containing 'relation' and 'relation_extraction_response'.
    - class_labels: A list of all possible class labels.

    Returns:
    - micro_f1: Micro F1 score.
    - micro_precision: Micro precision.
    - micro_recall: Micro recall.
    """
    with open(data_path, 'r') as file:
        data = json.load(file)
        
    class_labels = [record['relation'] for record in data]
    class_labels = list(set(class_labels))
    
    true_positives = {label: 0 for label in class_labels}
    false_positives = {label: 0 for label in class_labels}
    false_negatives = {label: 0 for label in class_labels}

    for record in data:
        true_label = record["relation"]
        predicted_label = record["relation_extraction_response"]

        for label in class_labels:
            if true_label == label and predicted_label == label:
                true_positives[label] += 1
            elif true_label == label:
                false_negatives[label] += 1
            elif predicted_label == label:
                false_positives[label] += 1

    # Calculate micro precision, recall, and F1 score
    total_true_positives = sum(true_positives.values())
    total_false_positives = sum(false_positives.values())
    total_false_negatives = sum(false_negatives.values())

    try:
        micro_precision = total_true_positives / (total_true_positives + total_false_positives)
    except ZeroDivisionError:
        micro_precision = 0.0

    try:
        micro_recall = total_true_positives / (total_true_positives + total_false_negatives)
    except ZeroDivisionError:
        micro_recall = 0.0

    try:
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
    except ZeroDivisionError:
        micro_f1 = 0.0

    macro_precision = sum(true_positives[label] / (true_positives[label] + false_positives[label])
                         if (true_positives[label] + false_positives[label]) > 0 else 0.0
                         for label in class_labels) / len(class_labels)

    macro_recall = sum(true_positives[label] / (true_positives[label] + false_negatives[label])
                      if (true_positives[label] + false_negatives[label]) > 0 else 0.0
                      for label in class_labels) / len(class_labels)

    try:
        macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
    except ZeroDivisionError:
        macro_f1 = 0.0

    return micro_f1, macro_f1, micro_precision, macro_precision, micro_recall, macro_recall


# Example usage:

data_path = './results/gpt_FewRel_sep_test_m=10.json'

micro_f1, macro_f1, micro_precision, macro_precision, micro_recall, macro_recall = calculate_manual_metrics(data_path)

print(f"Micro F1: {micro_f1}")
print(f"Micro Precision: {micro_precision}")
print(f"Micro Recall: {micro_recall}")
print(f"Macro F1: {macro_f1}")
print(f"Macro Precision: {macro_precision}")
print(f"Macro Recall: {macro_recall}")

