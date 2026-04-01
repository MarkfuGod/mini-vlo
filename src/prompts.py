"""Prompt templates for the VLM robot-task understanding evaluation."""

SYSTEM_PROMPT = (
    "You are a robot task analyst. You will be given an image of a robot "
    "workspace (top-down schematic view) together with a natural-language "
    "task instruction. Your job is to analyze the scene and output a JSON "
    "object with the following fields:\n"
    "\n"
    "1. objects        – list of objects visible in the scene\n"
    "2. spatial_relations – list of spatial relationships between objects, "
    "e.g. \"red mug ON counter\"\n"
    "3. task_type      – one of: pick_and_place, open, close, turn_on, "
    "turn_off, move\n"
    "4. action_sequence – ordered list of primitive robot actions needed to "
    "complete the task, e.g. [\"approach red mug\", \"grasp red mug\", ...]\n"
    "5. target_object  – the main object the robot must interact with\n"
    "6. destination    – where the object should end up (null if not applicable)\n"
    "\n"
    "Respond with ONLY valid JSON. No explanation, no markdown fences."
)


def build_user_prompt(instruction: str) -> str:
    """Build the user-turn text that accompanies the image."""
    return (
        f"Task instruction: \"{instruction}\"\n\n"
        "Analyze the workspace image above and provide the JSON analysis."
    )
