from pathlib import Path
import json

def get_qa_test_json(filename: str = "AAPL_qa.json") -> dict:
    current_dir = Path(__file__).parent
    json_path = current_dir.parent.parent / "resources" / "q_a_test" / filename
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

