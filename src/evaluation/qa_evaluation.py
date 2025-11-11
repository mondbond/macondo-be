from evaluate import load

from src.service.file_format_service import soup_html_to_text, any_format_to_str
from src.service.graph.core.subquery_retrieval_graph1 import \
  run_subquery_search_in_report_full_state
from src.service.query_report_service import \
  base_query_report_question_answer_full_state
from src.usecase.report_uc import save_text_report
from src.util.logger import logger

from src.evaluation.test_util import get_qa_test_json


with open("/Users/ibahr/Desktop/reports/AAPL.html", "rb") as f:
  pdf_content = f.read()
  # report = soup_html_to_text(pdf_content)
  report = any_format_to_str(pdf_content, "text/html")

  metadata = {
    "ticker": "AAPL",
    "date": "2025-07-17"
  }

  save_text_report(report, metadata)

bertscore = load("bertscore")

test_data = get_qa_test_json("AAPL_qa.json")

answers = []
questions = []
references = []

for item_map in test_data:
    logger.info(f"Evaluating question: {item_map['question']}")
    # result = run_subquery_search_in_report_full_state(ticker="AAPL", question=item_map['question'])['final_answer']
    result = base_query_report_question_answer_full_state(ticker="AAPL", query=item_map['question'])['answer']
    # answer = result.get("final_answer", "") if isinstance(result, dict) else str(result)
    answers.append(str(result))
    questions.append(item_map['question'])
    references.append(str(item_map['reference']))


results = bertscore.compute(
    predictions=answers,
    references=references,
    model_type="microsoft/deberta-xlarge-mnli",  # or "bert-base-uncased" for faster evaluation
    lang="en"
)

for i, q in enumerate(questions):
  logger.info(f"\nQuestion: {q}")
  logger.info(f"Prediction: {answers[i]}")
  logger.info(f"Reference: {references[i]}")
  logger.info(f"BERTScore F1: {results['f1'][i]:.4f}")

# Average score
avg_f1 = sum(results['f1']) / len(results['f1'])
logger.info(f"Average semantic similarity (BERTScore F1): {avg_f1:.4f}")
