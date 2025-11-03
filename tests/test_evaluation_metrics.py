import pytest
from collections import defaultdict

from src.models.router import UserIntentionEnum
from src.util.logger import logger
from src.evaluation.route_evaluation import calculate_classification_metrics
from src.service.graph.intention_service import classify_intent_with_prompt

test_list = [
  {
    "query": "What's the latest news on Apple Inc.?",
    "intention": UserIntentionEnum.NEWS_ABOUT_COMPANY.name
  },
  {
    "query": "Who is the main competitors for Microsoft company ticker MSFT based ot it's annual report?",
    "intention": UserIntentionEnum.COMPANY_INFORMATION_FROM_REPORT.name
  },
  {
    "query": "Why Tesla (ticker TSLA) fall yesterday?",
    "intention": UserIntentionEnum.ANALYSE_SHARE_PRISE.name
  },
  {
    "query": "What is the capital of France?",
    "intention": UserIntentionEnum.NOT_RELATED.name
  },
  {
    "query": "Give me the recent news about Amazon's market performance. ticker AMZN",
    "intention": UserIntentionEnum.NEWS_ABOUT_COMPANY.name
  },
  {
    "query": "What is the revenue of Google according to its latest report? ticker GOOGL",
    "intention": UserIntentionEnum.COMPANY_INFORMATION_FROM_REPORT.name
  },
  {
    "query": "Why did Facebook's stock drop yesterday? ticker META",
    "intention": UserIntentionEnum.ANALYSE_SHARE_PRISE.name
  },
  {
    "query": "Tell me a joke about programmers.",
    "intention": UserIntentionEnum.NOT_RELATED.name
  }

]

def calculate_classification_metrics_h(results):
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


def test_metrics_perfect_predictions():
    sample_results = [
        {'query': "What's the latest news on Apple Inc.?", 'predicted_intention': 'NEWS_ABOUT_COMPANY', 'expected_intention': 'NEWS_ABOUT_COMPANY', 'is_correct': True},
        {'query': 'Can you provide the financial report for Microsoft?', 'predicted_intention': 'COMPANY_INFORMATION_FROM_REPORT', 'expected_intention': 'COMPANY_INFORMATION_FROM_REPORT', 'is_correct': True},
        {'query': "How did Tesla's stock perform last week?", 'predicted_intention': 'ANALYSE_SHARE_PRISE', 'expected_intention': 'ANALYSE_SHARE_PRISE', 'is_correct': True},
        {'query': 'What is the capital of France?', 'predicted_intention': 'NOT_RELATED', 'expected_intention': 'NOT_RELATED', 'is_correct': True}
    ]
    metrics = calculate_classification_metrics(sample_results)
    assert metrics['accuracy'] == 1.0
    assert metrics['macro_recall'] == 1.0
    assert metrics['macro_precision'] == 1.0
    for recall in metrics['recall_per_intention'].values():
        assert recall == 1.0
    for precision in metrics['precision_per_intention'].values():
        assert precision == 1.0


def test_metrics_imperfect_predictions():
    sample_results = [
        {'query': "What's the latest news on Apple Inc.?", 'predicted_intention': 'NEWS_ABOUT_COMPANY', 'expected_intention': 'NEWS_ABOUT_COMPANY', 'is_correct': True},
        {'query': 'Can you provide the financial report for Microsoft?', 'predicted_intention': 'NEWS_ABOUT_COMPANY', 'expected_intention': 'COMPANY_INFORMATION_FROM_REPORT', 'is_correct': False},
        {'query': "How did Tesla's stock perform last week?", 'predicted_intention': 'ANALYSE_SHARE_PRISE', 'expected_intention': 'ANALYSE_SHARE_PRISE', 'is_correct': True},
        {'query': 'What is the capital of France?', 'predicted_intention': 'NOT_RELATED', 'expected_intention': 'NOT_RELATED', 'is_correct': True}
    ]
    metrics = calculate_classification_metrics(sample_results)
    assert metrics['accuracy'] == 0.75
    # Macro recall and precision should be less than 1.0 due to the error
    assert metrics['macro_recall'] < 1.0
    assert metrics['macro_precision'] < 1.0
    # Check that at least one recall/precision is less than 1.0
    assert any(r < 1.0 for r in metrics['recall_per_intention'].values())
    assert any(p < 1.0 for p in metrics['precision_per_intention'].values())

def test_evaluate_route_classification():
  results = []
  for test in test_list:
    state = {
      "user_message": test["query"],
      "history": []
    }
    user_intention = classify_intent_with_prompt(state)
    predicted_intention = user_intention.intention.name
    expected_intention = test["intention"]
    is_correct = str(predicted_intention) == str(expected_intention)
    results.append({
      "query": test["query"],
      "predicted_intention": predicted_intention,
      "expected_intention": expected_intention,
      "is_correct": is_correct
    })
  logger.info(f"Test {results}")
  summary_results = calculate_classification_metrics_h(results)
  logger.info(summary_results)
  assert summary_results["macro_recall"] >= 0.5
