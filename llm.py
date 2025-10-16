import re
import json
import base64
from enum import Enum
from pathlib import Path
from typing import List, Union, Tuple, Optional, Dict, Literal

from pydantic import BaseModel, Field
from config import anthropic_client, ANTHROPIC_MODEL

class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"

# --- PYDANTIC SCHEMAS ---
class ObjectiveDefinitionTool(BaseModel):
    """Tool to define the user's objective with a refined instruction and a quantitative goal."""
    refined_instruction: str = Field(..., description="A concise, actionable instruction for the AI agent, preserving the user's core keywords.")
    top_k: int = Field(..., description="The number of items to find or extract. Default to 3 if unspecified. Use 1 for requests about a single specific item.")

class MasterPlanTool(BaseModel):
    """Tool to define a high-level, multi-step plan to achieve the user's objective."""
    reasoning: str = Field(..., description="A step-by-step thought process explaining the generated plan.")
    plan: List[dict] = Field(..., description="A list of sequential goals. Each dictionary must have 'step_name', 'prerequisites', and 'goal_instructions'.")

class TargetingTool(BaseModel):
    """Tool to determine the specific element to find on the page for a given sub-goal."""
    goal_description: str = Field(..., description="A brief (2-3 word) description of the immediate goal, e.g., 'Close Pop-up', 'Find Search Bar'.")
    keywords: List[str] = Field(..., description="A list of case-sensitive text strings to search for on the page to find the target element(s).")
    box: Optional[List[int]] = Field(None, description="Optional. The bounding box `[x1, y1, x2, y2]` of the most important visual target on the screen.")

class TapTool(BaseModel):
    """Tool to tap/click a standard element like a button, link, or checkbox."""
    type: Literal["tap"] = "tap"
    selector: str = Field(..., description="The most robust XPath selector for the element to tap, chosen from the candidate elements list.")
    reason: str = Field(..., description="An explanation of why this element is being tapped.")

class NavigateTool(BaseModel):
    """Tool to navigate to a new page context WITHIN the site by clicking a link or profile element."""
    type: Literal["navigate"] = "navigate"
    selector: str = Field(..., description="The most robust XPath selector for the link or element that leads to the next page context.")
    reason: str = Field(..., description="An explanation of why navigating via this element is the correct next step.")

## --- UPGRADE --- ##
# Replaced FillAndSubmitTool with more granular FillTool and PressKeyTool for reliability.
class FillTool(BaseModel):
    """Tool to fill an input field with text."""
    type: Literal["fill"] = "fill"
    selector: str = Field(..., description="The most robust XPath selector for the input element.")
    text: str = Field(..., description="The text to be entered into the input field.")
    reason: str = Field(..., description="An explanation for this fill action.")

class PressKeyTool(BaseModel):
    """Tool to press a special key, like Enter, on a specific element."""
    type: Literal["press_key"] = "press_key"
    selector: str = Field(..., description="The XPath selector of the element to receive the key press (e.g., a search bar).")
    key: Literal["ENTER"] = Field(..., description="The special key to press. Currently only ENTER is supported.")
    reason: str = Field(..., description="An explanation for why this key press is necessary (e.g., 'To submit the search').")

class ScrollTool(BaseModel):
    """Tool to scroll the page or a specific scrollable element."""
    type: Literal["scroll"] = "scroll"
    reason: str = Field(..., description="An explanation of why scrolling is necessary.")

## --- UPGRADE --- ##
# Added a dedicated tool for handling pagination elements.
class PaginationTool(BaseModel):
    """Tool to click a 'Next Page' or page number button to load more results."""
    type: Literal["pagination"] = "pagination"
    selector: str = Field(..., description="The XPath selector for the 'Next Page' or numbered page link to click.")
    reason: str = Field(..., description="An explanation of why clicking this pagination element is the correct next step.")

class ExtractTool(BaseModel):
    """Tool to extract structured data by identifying repeating patterns on the page."""
    type: Literal["extract"] = "extract"
    reason: str = Field(..., description="Explanation of the data being extracted and the patterns identified.")
    container_selector: str = Field(..., description="The robust, class-based XPath selector for the repeating parent element that contains all information for a single item.")
    field_xpaths: Dict[str, str] = Field(..., description="A dictionary mapping data fields (e.g., 'product_title', 'price', 'rating') to their RELATIVE XPath selectors within the container. The key 'product_title' is mandatory.")

class FinishGoalTool(BaseModel):
    """Tool to signal that the current multi-step goal (e.g., APPLY_FILTER) is now fully complete."""
    type: Literal["finish_goal"] = "finish_goal"
    reason: str = Field(..., description="A summary of how the current goal was completed, confirming it is time to move to the next step in the master plan.")

class FinishTool(BaseModel):
    """Tool to successfully finish the entire task because all steps in the plan are complete."""
    type: Literal["finish"] = "finish"
    reason: str = Field(..., description="A summary of how the overall objective was completed.")

## --- UPGRADE --- ##
# New specialized tools for the pop-up handling pre-processing step.
class DismissTool(BaseModel):
    """Tool to dismiss an unwanted pop-up, cookie banner, or modal."""
    type: Literal["dismiss"] = "dismiss"
    selector: str = Field(..., description="The XPath selector for the element to click to dismiss the pop-up (e.g., 'Close', 'X', 'Accept').")
    reason: str = Field(..., description="An explanation of what is being dismissed and why.")

class NoPopupTool(BaseModel):
    """Tool to signal that no pop-ups were found and it's clear to proceed with the main task."""
    type: Literal["no_popup"] = "no_popup"
    reason: str = Field(..., description="A confirmation that the page is clear of any interruptions.")

class MultiActionTool(BaseModel):
    """A tool to execute a sequence of simple actions to achieve a single logical goal, like performing a search."""
    type: Literal["multi_action"] = "multi_action"
    reason: str = Field(..., description="A summary of the overall goal that this action chain achieves.")
    # Updated the Union to include the new granular form tools.
    actions: List[Union[TapTool, FillTool, PressKeyTool]] = Field(..., description="A list of actions to execute in sequence.")

class ConfirmActionTool(BaseModel):
    """Tool to confirm and execute a simple, high-confidence action like a tap or click."""
    type: Literal["confirm_action"] = "confirm_action"
    selector: str = Field(..., description="The XPath selector of the visually confirmed element to tap.")
    reason: str = Field(..., description="A brief confirmation that this is the correct action to proceed with the current task.")

# Updated tool lists
POPUP_CHECK_TOOLS = [DismissTool, NoPopupTool]
SLOW_BRAIN_TOOLS = [MultiActionTool, TapTool, FillTool, PressKeyTool, ScrollTool, NavigateTool, PaginationTool, ExtractTool, FinishGoalTool, FinishTool]
FAST_BRAIN_TOOLS = [ConfirmActionTool]

# --- PROMPT TEMPLATES ---
REFINER_PROMPT = """
You are an Objective Parser. Your job is to analyze the user's request and transform it into a structured objective for an AI web agent. You must call the `ObjectiveDefinitionTool`.

**User's Target URL:** {url}
**User's Query:** "{query}"

**Your Rules:**
1.  **Extract `top_k`:** Analyze the query to find the number of items the user wants (e.g., "top 3", "best 5"). This is your `top_k`.
    - If a specific number is mentioned, use it.
    - If the user asks for a single specific thing (e.g., "the price of the iPhone 15"), set `top_k` to 1.
    - If no number is mentioned, you MUST default `top_k` to 3.
2.  **Refine Instruction:** Create a clear, actionable instruction for the agent in the `refined_instruction` field.
    - **CRITICAL:** You must preserve the user's original core keywords and intent. Do NOT add extra words like "supplements" if the user only asked for "whey protein". Clarify the goal, do not change the subject.

Call the `ObjectiveDefinitionTool` with your parsed results.
"""

PLANNER_PROMPT = """
You are the "Planner" module for an autonomous web agent. Your critical task is to decompose the user's objective into a structured, sequential plan.

**User's Objective:** "{query}"

**CRITICAL CONTEXT:** The agent has already successfully navigated to the initial URL. Your plan should start with the first action required ON THE PAGE.

**PLANNING STYLE:** Think like a human.
- A search action should be a single step (`EXECUTE_SEARCH`).
- Recognize that some actions may require two sub-steps (e.g., clicking a search icon, then typing in a new field). Your instructions should guide the agent through this.

**CRITICAL HEURISTIC:** Do NOT add `APPLY_SORT` or `APPLY_FILTER` steps unless the user's query *explicitly* mentions sorting or filtering criteria.

**Your Task:**
1.  Analyze the user's objective, obeying the context and heuristics above.
2.  **Provide Human-Like Hints:**
    - For complex, multi-action tasks like `APPLY_FILTER`, your `goal_instructions` must provide contextual guidance. For a filter goal like 'ratings more than 4 stars', your instructions should be: "First, locate and open the main 'Filter' options. Then, within the filter panel that appears, find the 'Customer Ratings' section and select the '4★ & above' option."
    ## --- UPGRADE --- ##
    # Updated instructions to guide the LLM towards more robust, class-based selectors.
    - **For `EXTRACT_DATA` goals, your instructions must guide the agent on the modern extraction workflow:** "Analyze the page to identify the repeating HTML container for a single item. Prefer robust, class-based selectors (e.g., `//div[contains(@class, 'product-card')]`) over fragile, absolute ones. Your goal is to provide the XPath selector for this container and the relative XPaths for the data fields inside it (like product_title, price, rating). The code will then perform the fast extraction."
3.  Create a plan using valid goals: 'LOGIN', 'EXECUTE_SEARCH', 'APPLY_FILTER', 'APPLY_SORT', 'NAVIGATE', 'EXTRACT_DATA'.
4.  Call the `MasterPlanTool` with your reasoning and the final plan.
"""

## --- UPGRADE --- ##
# New, specialized prompt for the pop-up detection node.
POPUP_PROMPT = """
You are a "Pop-up Killer" for a web agent. Your only job is to analyze the screenshot for any kind of overlay, modal, or banner that interrupts the user's main task.

**Your Task:**
1.  Examine the screenshot for any element that obstructs the main page content. This includes, but is not limited to:
    - Cookie consent banners
    - Sign-in or registration modals
    - App download prompts
    - Notification permission dialogs
    - Subscription requests
    - "Rate our app" dialogs
    - Any other overlay that requires dismissal.
2.  **If such an element exists:** Find the button to dismiss it. Look for text like "Accept", "Continue", "Close", "No, thanks", "Maybe later", or an "X" icon. Call the `DismissTool` with the XPath selector of that dismissal button.
3.  **If the page is clear:** If there are no pop-ups or obstructive overlays, you MUST call the `NoPopupTool` to confirm it is safe to proceed.

This is a critical pre-processing step. Do not attempt to perform any other actions. Your decision determines if the agent can continue its main task.
"""

TARGETING_PROMPT = """
You are the "Targeting" module. Your job is to analyze the screenshot to determine what to look for to accomplish the current task.

**Current Task:** "{current_task}"
**Instructions for this task:** "{goal_instructions}"
**Current Sub-Step:** You are on sub-step {sub_step} of this task.
**Recent Action History:**
{history}

**Your Task:**
1.  **Analyze Context:** Based on the current sub-step, determine which part of the `goal_instructions` is most relevant now. If `sub_step` is 0, focus on the first part. If it's 1, focus on the second part.
2.  **Analyze the Screenshot:** Visually identify the most important element on the page needed to achieve the **current sub-step** of your task.
3.  **Extract Keywords:** Determine the text keywords associated with that element.
4.  **Provide Bounding Box:** If you are confident in your visual target, provide its `[x1, y1, x2, y2]` bounding box coordinates. This is critical for grounding the search.

Call the `TargetingTool` with your decision.
"""

FAST_AGENT_PROMPT = """
You are the "Fast Action" module. Your job is to perform a sanity check on a high-confidence action. The code has already visually confirmed a single best element to achieve the task. Your only job is to call the `ConfirmActionTool`.

**Current Task:** "{current_task}"
**Visually Confirmed Element:**
{confirmed_element}

**Your Task:**
1.  Briefly reason that this element is the correct one for the current task.
2.  Call the `ConfirmActionTool` with the element's selector. Do not change the selector.
"""

## --- UPGRADE --- ##
# The main agent prompt is heavily upgraded for form-filling and pagination.
AGENT_PROMPT = """
You are the "Action" module. You are given a current task, specific instructions, a list of candidate elements, and your recent history. Your job is to choose the single best action OR a chain of simple actions to progress on your task.

**User's Objective:** "{query}"
**Current Task:** "{current_task}"
**Instructions for this Task:** "{goal_instructions}"
**Current Sub-Step:** You are on sub-step {sub_step} of this task.
**Extraction Goal:** Find a total of `{top_k}` items.
**Extraction Progress:** You have already found `{results_count}` items.
**Stagnation Counter:** `{stagnation_count}`
**Recent Action History:**
{history}

**Candidate Elements Found on Page:**
{candidate_elements}

**Chain of Command (Follow in order):**
1.  **Stagnation Protocol:** If `Stagnation Counter` > 1, you have already tried scrolling and it yielded no new results. Your ONLY valid actions are to look for a "Next Page" or page number element and use the `PaginationTool`, OR use `FinishGoalTool` if you believe you are done. Do not `ScrollTool` again. If `Stagnation Counter` is 1, `ScrollTool` is a good option.
2.  **Action Chaining (CRITICAL):** For tasks requiring a sequence, you MUST use the `MultiActionTool`.
    - **Example Search Flow:** A search is NOT a single action. It's a chain: `TapTool` on the search bar/icon -> `FillTool` with the search query -> `PressKeyTool` with "ENTER" to submit.
    - You must use `MultiActionTool` to chain these `TapTool`, `FillTool`, and `PressKeyTool` actions together for any form interaction.
3.  **Task Adherence:** Your action(s) MUST be directly related to achieving the current `sub_step` of the `Current Task`.
4.  **Quantitative Goal Check:** If `Current Task` is `EXTRACT_DATA` and `results_count` >= `top_k`, your ONLY action is `FinishGoalTool`.
5.  **Multi-Step Execution:** Execute each part of the instruction sequentially, guided by the `sub_step` number. Only call `FinishGoalTool` once all sub-steps are complete.
6.  **Extraction Workflow:** For `EXTRACT_DATA` tasks, your job is to analyze the page and call the `ExtractTool`.
    - **CRITICAL:** Your `container_selector` must be a robust, class-based XPath that identifies the repeating parent element for each item. Avoid using IDs or highly specific, fragile selectors.
    - **IGNORE SPONSORED:** You must visually inspect the page for any signs of advertisements or sponsored content (e.g., text like "Sponsored", "Ad"). The code will perform a final check, but you are the first line of defense. Do not include selectors for items that appear to be ads.
7.  **Primary Action:** Choose the best tool (`MultiActionTool`, `TapTool`, etc.) to make progress. Select the most relevant candidate(s).
8.  **Fallback:** If no candidates are suitable, your only permitted actions are `ScrollTool` or `FinishTool`.
"""

# --- LLM HELPER FUNCTIONS ---
def get_llm_response(system_prompt: str, prompt: str, provider: LLMProvider, tools: List[BaseModel], images: List[Path] = []) -> Tuple[Optional[str], Optional[dict], dict]:
    # This function remains robust.
    usage = {"input_tokens": 0, "output_tokens": 0}
    try:
        if provider == LLMProvider.ANTHROPIC:
            if not anthropic_client: raise ValueError("Anthropic client not initialized.")
            tool_definitions = [{"name": t.model_json_schema()['title'], "description": t.model_json_schema().get('description'), "input_schema": t.model_json_schema()} for t in tools]
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            if images and images[0].is_file():
                with open(images[0], "rb") as f: img_data = base64.b64encode(f.read()).decode("utf-8")
                messages[0]["content"].append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_data}})
            response = anthropic_client.messages.create(model=ANTHROPIC_MODEL, max_tokens=2048, system=system_prompt, messages=messages, tools=tool_definitions, tool_choice={"type": "auto"})
            usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_name_from_api, tool_input = content_block.name, content_block.input
                    original_tool = next((t for t in tools if t.model_json_schema()['title'] == tool_name_from_api), None)
                    if original_tool: return original_tool.__name__, tool_input, usage
            return None, None, usage
    except Exception as e:
        print(f"LLM API call failed for {provider}: {e}")
        return FinishTool.__name__, {"reason": f"LLM API Error: {e}"}, usage
    return None, None, usage

def get_objective_definition(url: str, query: str, provider: LLMProvider) -> Tuple[dict, dict]:
    # This function is unchanged.
    prompt = REFINER_PROMPT.format(url=url, query=query)
    system_prompt = "You are an Objective Parser. Analyze the user's query and call the ObjectiveDefinitionTool."
    tool_name, tool_input, usage = get_llm_response(system_prompt, prompt, provider, tools=[ObjectiveDefinitionTool])
    if tool_name == ObjectiveDefinitionTool.__name__:
        return tool_input, usage
    return {"refined_instruction": query, "top_k": 3}, usage

def create_master_plan(query: str, provider: LLMProvider) -> Tuple[dict, dict]:
    # This function is unchanged.
    prompt = PLANNER_PROMPT.format(query=query)
    system_prompt = "You are the Planner module. Decompose the user's objective and call the MasterPlanTool."
    tool_name, tool_input, usage = get_llm_response(system_prompt, prompt, provider, tools=[MasterPlanTool])
    if tool_name == MasterPlanTool.__name__: return tool_input, usage
    return {"reasoning": "Failed to create a plan.", "plan": [{"step_name": "EXECUTE_SEARCH", "prerequisites": [], "goal_instructions": f"Execute a search based on the user's query: {query}"}]}, usage

## --- UPGRADE --- ##
# New function to specifically handle the pop-up detection logic.
def get_popup_decision(provider: LLMProvider, screenshot_path: Path) -> Tuple[dict, dict]:
    system_prompt = "You are a Pop-up Killer. Analyze the screenshot for interruptions and call the appropriate tool."
    tool_name, tool_input, usage = get_llm_response(system_prompt, POPUP_PROMPT, provider, tools=POPUP_CHECK_TOOLS, images=[screenshot_path])
    if tool_name and tool_input:
        # Normalize the tool name into a 'type' field for consistent processing in the graph.
        tool_input['type'] = re.sub(r'(?<!^)(?=[A-Z])', '_', tool_name).lower().replace("_tool","")
        return tool_input, usage
    # Default to assuming no pop-up if the LLM fails.
    return {"type": "no_popup", "reason": "LLM failed to respond, assuming no pop-up."}, usage


def get_targeting_decision(current_task: str, goal_instructions: str, history: str, provider: LLMProvider, screenshot_path: Path, sub_step: int) -> Tuple[dict, dict]:
    prompt = TARGETING_PROMPT.format(current_task=current_task, goal_instructions=goal_instructions, history=history, sub_step=sub_step)
    system_prompt = "You are the Targeting module. Identify the immediate goal, keywords, and bounding box, then call the TargetingTool."
    tool_name, tool_input, usage = get_llm_response(system_prompt, prompt, provider, tools=[TargetingTool], images=[screenshot_path])
    if tool_name == TargetingTool.__name__: return tool_input, usage
    return {"goal_description": "Error in targeting", "keywords": []}, usage

def get_fast_agent_action(current_task: str, confirmed_element: str, provider: LLMProvider) -> Tuple[dict, dict]:
    # This function remains for the fast path.
    prompt = FAST_AGENT_PROMPT.format(current_task=current_task, confirmed_element=confirmed_element)
    system_prompt = "You are the Fast Action module. Sanity check the provided element and call ConfirmActionTool."
    tool_name, tool_input, usage = get_llm_response(system_prompt, prompt, provider, tools=FAST_BRAIN_TOOLS)
    if tool_name and tool_input:
        return tool_input, usage
    return {"type": "finish", "reason": "Fast agent could not decide on a valid action."}, usage

def get_agent_action(query: str, current_task: str, goal_instructions: str, candidate_elements: str, history: str, provider: LLMProvider, screenshot_path: Path, top_k: int, results_count: int, stagnation_count: int, sub_step: int) -> Tuple[dict, dict]:
    prompt = AGENT_PROMPT.format(
        query=query, current_task=current_task, goal_instructions=goal_instructions,
        candidate_elements=candidate_elements, history=history, top_k=top_k, results_count=results_count,
        stagnation_count=stagnation_count, sub_step=sub_step
    )
    system_prompt = "You are the Action module. Choose the best tool, potentially chaining actions with MultiActionTool."
    tool_name, tool_input, usage = get_llm_response(system_prompt, prompt, provider, tools=SLOW_BRAIN_TOOLS, images=[screenshot_path])
    if tool_name and tool_input:
        # If the LLM returns a single action, wrap it for consistent processing.
        if tool_name != MultiActionTool.__name__:
             tool_input['type'] = re.sub(r'(?<!^)(?=[A-Z])', '_', tool_name).lower().replace("_tool","")
             return tool_input, usage
        return tool_input, usage
    return {"type": "finish", "reason": "Agent could not decide on a valid action."}, usage