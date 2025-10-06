import yaml
from pathlib import Path
from string import Template
from typing import Dict, Any, Optional


class PromptManager:
  def __init__(self, prompts_dir: str = "../../resources/prompts"):
    self.prompts_dir = Path(prompts_dir)
    self.prompts_map: Dict[str, Dict[str, Any]] = {}
    self._load_all_prompts()

  def _load_all_prompts(self):
    """Recursively load all YAML prompt files under prompts_dir."""
    yaml_files = list(self.prompts_dir.rglob("*.yml"))
    for file_path in yaml_files:
      with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        prompt_name = data.get("prompt_name")
        if not prompt_name:
          continue  # skip invalid files
        versions = data.get("versions", [])
        if not versions:
          continue
        # store in memory
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
    """Return prompt text by name, optionally by version or alias, with variable substitution."""
    prompt_entry = self.prompts_map.get(prompt_name)
    if not prompt_entry:
      raise ValueError(f"Prompt '{prompt_name}' not found")

    versions = prompt_entry["versions"]

    # select version by alias or number, or use default/latest
    prompt_data = None
    if alias:
      prompt_data = next((v for v in versions if v.get("alias") == alias), None)
    if not prompt_data and version is not None:
      prompt_data = next((v for v in versions if v.get("version") == version), None)
    if not prompt_data:
      # fallback to default/latest
      default_version = prompt_entry.get("default_version", versions[-1]["version"])
      prompt_data = next((v for v in versions if v.get("version") == default_version), versions[-1])

    template_text = prompt_data["template"]

    return template_text

  def list_prompts(self):
    """Return list of all loaded prompt names."""
    return list(self.prompts_map.keys())

  def list_versions(self, prompt_name: str):
    """Return all versions (with alias and description) for a prompt."""
    prompt_entry = self.prompts_map.get(prompt_name)
    if not prompt_entry:
      return []
    return [{"version": v["version"], "alias": v.get("alias"), "description": v.get("description")} for v in prompt_entry["versions"]]


if __name__ == "__main__":
  pm = PromptManager()
  print(pm.list_prompts())
  prompt_text = pm.get_prompt("context_based_answer")
  print(prompt_text)
  print(pm.list_versions("example_prompt"))
