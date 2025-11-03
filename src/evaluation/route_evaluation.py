from src.service.graph.intention_service import classify_intent_with_prompt
from src.util.logger import logger



def evaluate_route_classification(classify_intent_func):
    results = []
    for test in test_list:
        state = {
            "user_message": test["query"],
            "history": []
        }
        user_intention = classify_intent_func(state)
        predicted_intention = user_intention.intention.name
        expected_intention = test["intention"]
        is_correct = str(predicted_intention) == str(expected_intention)
        results.append({
            "query": test["query"],
            "predicted_intention": predicted_intention,
            "expected_intention": expected_intention,
            "is_correct": is_correct
        })
    return results

def calculate_classification_metrics(results):
    from collections import defaultdict
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    accuracy = correct / total if total else 0

    # Collect all unique intentions
    intentions = {r['expected_intention'] for r in results} | {r['predicted_intention'] for r in results}
    # Initialize counters
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)

    for r in results:
        pred = r['predicted_intention']
        exp = r['expected_intention']
        support[exp] += 1
        if pred == exp:
            tp[exp] += 1
        else:
            fp[pred] += 1
            fn[exp] += 1

    # Calculate recall and precision for each intention
    recall_per_intention = {}
    precision_per_intention = {}
    for intent in intentions:
        recall = tp[intent] / (tp[intent] + fn[intent]) if (tp[intent] + fn[intent]) else 0
        precision = tp[intent] / (tp[intent] + fp[intent]) if (tp[intent] + fp[intent]) else 0
        recall_per_intention[intent] = recall
        precision_per_intention[intent] = precision

    # Macro averages
    macro_recall = sum(recall_per_intention.values()) / len(intentions) if intentions else 0
    macro_precision = sum(precision_per_intention.values()) / len(intentions) if intentions else 0

    return {
        'accuracy': accuracy,
        'macro_recall': macro_recall,
        'macro_precision': macro_precision,
        'recall_per_intention': recall_per_intention,
        'precision_per_intention': precision_per_intention,
        'support': dict(support)
    }


if __name__ == "__main__":
  results = evaluate_route_classification(classify_intent_with_prompt)
  logger.info(results)
  metrics = calculate_classification_metrics(results)
  logger.info(metrics)
