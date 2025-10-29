import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class PromptManager:
  def __init__(self, prompts_dir: str = "resources/prompts"):
    # Resolve path relative to this file if not absolute
    base_dir = Path(__file__).parent.parent.parent  # src/util -> project root
    prompts_path = Path(prompts_dir)
    if not prompts_path.is_absolute():
      prompts_path = base_dir / prompts_path
    self.prompts_dir = prompts_path.resolve()
    self.prompts_map: Dict[str, Dict[str, Any]] = {}
    self._load_all_prompts()

  def _load_all_prompts(self):
    yaml_files = list(self.prompts_dir.rglob("*.yml"))
    for file_path in yaml_files:
      with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        prompt_name = data.get("prompt_name")
        if not prompt_name:
          continue
        versions = data.get("versions", [])
        if not versions:
          continue
        self.prompts_map[prompt_name] = {
          "versions": versions,
          "default_version": data.get("default_version", versions[-1]["version"])
        }

  def get_prompt(
      self,
      prompt_name: str,
      version: Optional[int] = None,
      alias: Optional[str] = None,
  ) -> str:
    prompt_entry = self.prompts_map.get(prompt_name)
    if not prompt_entry:
      raise ValueError(f"Prompt '{prompt_name}' not found")

    versions = prompt_entry["versions"]

    prompt_data = None
    if alias:
      prompt_data = next((v for v in versions if v.get("alias") == alias), None)
    if not prompt_data and version is not None:
      prompt_data = next((v for v in versions if v.get("version") == version), None)
    if not prompt_data:
      default_version = prompt_entry.get("default_version", versions[-1]["version"])
      prompt_data = next((v for v in versions if v.get("version") == default_version), versions[-1])

    template_text = prompt_data["template"]

    return template_text

  def list_prompts(self):
    return list(self.prompts_map.keys())

  def list_versions(self, prompt_name: str):
    prompt_entry = self.prompts_map.get(prompt_name)
    if not prompt_entry:
      return []
    return [{"version": v["version"], "alias": v.get("alias"), "description": v.get("description")} for v in prompt_entry["versions"]]

prompt_manager = PromptManager()


if __name__ == "__main__":
  pm = PromptManager()
  print(pm.list_prompts())
  prompt_text = pm.get_prompt("context_based_answer")
  print(prompt_text)
  print(pm.list_versions("example_prompt"))
