import asyncio
import uuid
import json
import time
import csv
import re
import sqlite3
from pathlib import Path
from urllib.parse import urljoin
import traceback
from typing import List, TypedDict, Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from appium import webdriver
from appium.options.android import UiAutomator2Options
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException, StaleElementReferenceException

## --- UPGRADE --- ##
# Import the new pop-up decision function
from llm import LLMProvider, get_objective_definition, create_master_plan, get_targeting_decision, get_agent_action, get_fast_agent_action, get_popup_decision
from config import SCREENSHOTS_DIR, ANTHROPIC_MODEL

# --- SETUP AND UTILITIES (Unchanged) ---
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)
REPORT_CSV_FILE = Path("report.csv")
app = FastAPI(title="LangGraph Android Web Agent")
JOB_QUEUES, JOB_RESULTS = {}, {}
TOKEN_COSTS = { "anthropic": { "claude-3-5-sonnet-20240620": {"input": 3.0, "output": 15.0} } }
MODEL_MAPPING = { LLMProvider.ANTHROPIC: ANTHROPIC_MODEL }
APPIUM_SERVER_URL = "http://localhost:4723"

def get_current_timestamp(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def push_status(job_id: str, msg: str, details: dict = None):
    q = JOB_QUEUES.get(job_id)
    if q: q.put_nowait({"ts": get_current_timestamp(), "msg": msg, **({"details": details} if details else {})})

def resize_image_if_needed(path: Path):
    try:
        with Image.open(path) as img:
            if max(img.size) > 2000: img.thumbnail((2000, 2000), Image.LANCZOS); img.save(path)
    except Exception as e: print(f"Warning: Could not resize {path}. Error: {e}")

def save_analysis_report(run_dir: Path, data: dict):
    # This function is unchanged
    job_id = data["job_id"]; token_usage = data.get("token_usage", [])
    total_input = sum(s.get("input_tokens", 0) for s in token_usage); total_output = sum(s.get("output_tokens", 0) for s in token_usage)
    data.update({"total_input_tokens": total_input, "total_output_tokens": total_output})
    provider, model = data.get("provider", "anthropic"), data.get("model", ANTHROPIC_MODEL)
    cost_key = next((k for k in TOKEN_COSTS.get(provider, {}) if model.lower() in k.lower()), None)
    cost_info = TOKEN_COSTS.get(provider, {}).get(cost_key)
    total_cost = ((total_input / 1e6) * cost_info["input"] + (total_output / 1e6) * cost_info["output"]) if cost_info else 0.0
    data["total_cost_usd"] = f"{total_cost:.5f}"
    try:
        with open(run_dir / "analysis_report.json", 'w') as f: json.dump(data, f, indent=2)
        file_exists = REPORT_CSV_FILE.is_file()
        with open(REPORT_CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(['job_id', 'total_input_tokens', 'total_output_tokens', 'total_cost_usd'])
            writer.writerow([job_id, total_input, total_output, data["total_cost_usd"]])
    except Exception as e: print(f"Error saving analysis: {e}")

class SearchRequest(BaseModel):
    url: str; query: str; llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    desktop_mode: bool = False

class AgentState(TypedDict):
    job_id: str; driver: webdriver.Remote; query: str; provider: LLMProvider; desktop_mode: bool
    refined_query: str; top_k: int; results: List[dict]; run_dir: Path
    step: int; max_steps: int; last_action: dict; history: List[str]; token_usage: List[dict]
    master_plan: List[dict]
    plan_step: int
    sub_step: int
    stagnation_count: int
    candidate_elements: List[dict]
    confirmed_candidate: Optional[dict]
    ## --- UPGRADE --- ##
    # Added state to track the current phase (pop-up check vs. main task).
    current_phase: Literal["popup_check", "main_task"]


def get_absolute_xpath(driver: webdriver.Remote, element: WebElement) -> str:
    # This function is unchanged
    return driver.execute_script(
        "return (function getXPath(element) {"
        "if (element.id !== '') return `//*[@id=\"${element.id}\"]`;"
        "if (element === document.body) return element.tagName.toLowerCase();"
        "let ix = 0;"
        "const siblings = element.parentNode.childNodes;"
        "for (let i = 0; i < siblings.length; i++) {"
        "const sibling = siblings[i];"
        "if (sibling === element) return `${getXPath(element.parentNode)}/${element.tagName.toLowerCase()}[${ix + 1}]`;"
        "if (sibling.nodeType === 1 && sibling.tagName === element.tagName) ix++;"
        "}"
        "})(arguments[0]);", element)

def find_candidate_elements(driver: webdriver.Remote, keywords: List[str], box: Optional[List[int]] = None) -> List[dict]:
    # This function's logic remains robust.
    if not keywords: return []
    queries = []
    for k in keywords:
        k_escaped = k.replace("'", "\"'\"")
        queries.extend([f"contains(normalize-space(.), '{k_escaped}')", f"contains(@placeholder, '{k_escaped}')", f"contains(@aria-label, '{k_escaped}')", f"contains(@title, '{k_escaped}')"])
    full_query = " or ".join(queries)
    xpath_targeted = f"//*[self::a or self::button or self::input or self::textarea or @role='button' or @onclick][{full_query}]"
    found_elements = []
    try:
        found_elements.extend(driver.find_elements(By.XPATH, xpath_targeted))
        if len(found_elements) < 15:
            xpath_fallback = f"//*[self::div or self::span or self::p][{full_query}]"
            found_elements.extend(driver.find_elements(By.XPATH, xpath_fallback))
    except Exception as e: print(f"Error during element search: {e}")
    candidates = []
    for element in found_elements:
        try:
            if not element.is_displayed() or not element.is_enabled(): continue
            tag = element.tag_name.lower()
            text = (element.text or element.get_attribute("value") or element.get_attribute("aria-label") or "").strip()
            selector = f"//{get_absolute_xpath(driver, element)[2:]}"
            score = 10 if tag in ['button', 'a', 'input'] else 5
            if 'sponsored' in text.lower() or 'ad' in text.lower(): score -= 5
            loc, size = element.location, element.size
            el_box = [loc['x'], loc['y'], loc['x'] + size['width'], loc['y'] + size['height']]
            candidates.append({"score": score, "selector": selector, "tag": tag, "text": text[:100], "box": el_box})
        except StaleElementReferenceException: continue
        except Exception: pass
    if box and candidates:
        for cand in candidates:
            x_left, y_top = max(cand['box'][0], box[0]), max(cand['box'][1], box[1])
            x_right, y_bottom = min(cand['box'][2], box[2]), min(cand['box'][3], box[3])
            if x_right < x_left or y_bottom < y_top:
                cand['overlap'] = 0.0; continue
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            cand_area = (cand['box'][2] - cand['box'][0]) * (cand['box'][3] - cand['box'][1])
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            cand['overlap'] = intersection_area / float(cand_area + box_area - intersection_area)
        return sorted(candidates, key=lambda x: x.get('overlap', 0.0), reverse=True)
    return sorted(candidates, key=lambda x: x['score'], reverse=True)[:15]

# --- GRAPH NODES ---
def navigate_to_page(state: AgentState) -> AgentState:
    # This node remains robust.
    state['driver'].get(state['query'])
    push_status(state['job_id'], "page_navigation_started", {"url": state['query']})
    try: WebDriverWait(state['driver'], 15).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    except TimeoutException: print("Warning: Page body did not load in 15 seconds.")
    push_status(state['job_id'], "page_load_complete", {"url": state['driver'].current_url})
    return state

def planner_node(state: AgentState) -> AgentState:
    # This node remains robust.
    push_status(state['job_id'], "agent_planning")
    plan_data, usage = create_master_plan(state['refined_query'], state['provider'])
    state['master_plan'] = plan_data.get('plan', [])
    state['plan_step'] = 0
    state['sub_step'] = 0
    state['token_usage'].append({"task": "planner", **usage})
    if not state['master_plan']:
        state['last_action'] = {"type": "finish", "reason": "Master plan could not be created."}
    push_status(state['job_id'], "agent_plan_created", {"plan": state['master_plan'], "reasoning": plan_data.get('reasoning'), "usage": usage})
    return state

## --- UPGRADE --- ##
# New node dedicated to detecting and dismissing pop-ups before the main logic.
def dismiss_popups_node(state: AgentState) -> AgentState:
    state['current_phase'] = "popup_check"
    push_status(state['job_id'], "popup_check_started")
    screenshot_path = state['run_dir'] / "screenshots" / f"popup_check_{state['step']:02d}.png"
    state['driver'].get_screenshot_as_file(str(screenshot_path))
    resize_image_if_needed(screenshot_path)

    action, usage = get_popup_decision(state['provider'], screenshot_path)
    state['token_usage'].append({"task": f"popup_check_step_{state['step']}", **usage})
    push_status(state['job_id'], "popup_check_decision", {"decision": action, "usage": usage})

    state['last_action'] = action
    return state

def targeting_node(state: AgentState) -> AgentState:
    # This node now sets the phase to 'main_task'.
    state['current_phase'] = "main_task"
    job_id, driver = state['job_id'], state['driver']
    plan, plan_step = state['master_plan'], state['plan_step']
    current_task = plan[plan_step]['step_name']
    goal_instructions = plan[plan_step]['goal_instructions']
    push_status(job_id, "agent_step", {"step": state['step'], "max_steps": state['max_steps'], "current_task": current_task, "instructions": goal_instructions, "sub_step": state['sub_step'], "top_k": state['top_k'], "results_found": len(state['results'])})
    screenshot_path = state['run_dir'] / "screenshots" / f"{state['step']:02d}_step.png"
    driver.get_screenshot_as_file(str(screenshot_path)); resize_image_if_needed(screenshot_path)
    history_str = "\n".join(state['history'])
    targeting, usage = get_targeting_decision(current_task=current_task, goal_instructions=goal_instructions, history=history_str, provider=state['provider'], screenshot_path=screenshot_path, sub_step=state['sub_step'])
    state['token_usage'].append({"task": f"targeting_step_{state['step']}", **usage})
    push_status(job_id, "agent_targeting", {"decision": targeting, "usage": usage})
    candidates = find_candidate_elements(driver, targeting.get("keywords", []), targeting.get("box"))
    state['candidate_elements'] = candidates
    if candidates and candidates[0].get('overlap', 0.0) > 0.9:
        state['confirmed_candidate'] = candidates[0]
    else:
        state['confirmed_candidate'] = None
    return state

def fast_reasoning_node(state: AgentState) -> AgentState:
    # The fast brain remains for high-confidence, single actions.
    action, usage = get_fast_agent_action(
        current_task=state['master_plan'][state['plan_step']]['step_name'],
        confirmed_element=json.dumps(state['confirmed_candidate'], indent=2),
        provider=state['provider']
    )
    state['token_usage'].append({"task": f"fast_action_step_{state['step']}", **usage})
    push_status(state['job_id'], "agent_action_thought", {"thought": action.get("reason"), "usage": usage, "brain": "fast"})
    state['last_action'] = action
    return state

def slow_reasoning_node(state: AgentState) -> AgentState:
    # The slow brain handles complex reasoning and action chaining.
    screenshot_path = state['run_dir'] / "screenshots" / f"{state['step']:02d}_step.png"
    candidates_str = "No suitable elements found." if not state['candidate_elements'] else json.dumps(state['candidate_elements'], indent=2)
    action, usage = get_agent_action(
        query=state['refined_query'],
        current_task=state['master_plan'][state['plan_step']]['step_name'],
        goal_instructions=state['master_plan'][state['plan_step']]['goal_instructions'],
        candidate_elements=candidates_str,
        history="\n".join(state['history']),
        provider=state['provider'],
        screenshot_path=screenshot_path,
        top_k=state['top_k'], results_count=len(state['results']),
        stagnation_count=state['stagnation_count'], sub_step=state['sub_step']
    )
    state['token_usage'].append({"task": f"slow_action_step_{state['step']}", **usage})
    push_status(state['job_id'], "agent_action_thought", {"thought": action.get("reason"), "usage": usage, "brain": "slow"})
    state['last_action'] = action
    return state

def execute_action_node(state: AgentState) -> AgentState:
    ## --- UPGRADE --- ##
    # This node is heavily upgraded to handle new granular actions and multi-action chains.
    job_id, driver = state['job_id'], state['driver']
    actions = state['last_action'].get('actions', [state['last_action']])
    for i, action in enumerate(actions):
        push_status(job_id, "executing_action", {"action": action, "chain_step": f"{i+1}/{len(actions)}"})
        action_type = action.get("type")
        action_succeeded = False
        error_message = ""
        try:
            if action_type in ["tap", "navigate", "confirm_action", "dismiss", "pagination"]:
                wait = WebDriverWait(driver, 10); selector = action.get("selector")
                element = wait.until(EC.element_to_be_clickable((By.XPATH, selector)))
                try: element.click(); action_succeeded = True
                except Exception:
                    try: driver.execute_script("arguments[0].click();", element); action_succeeded = True
                    except Exception:
                        try: ActionChains(driver).move_to_element(element).click().perform(); action_succeeded = True
                        except Exception as e3: error_message = str(e3).splitlines()[0]

            elif action_type == "fill":
                wait = WebDriverWait(driver, 10); selector = action.get("selector")
                element = wait.until(EC.element_to_be_clickable((By.XPATH, selector)))
                element.click(); element.clear(); element.send_keys(action["text"])
                action_succeeded = True
            
            elif action_type == "press_key":
                wait = WebDriverWait(driver, 10); selector = action.get("selector")
                element = wait.until(EC.element_to_be_clickable((By.XPATH, selector)))
                element.send_keys(Keys.ENTER)
                action_succeeded = True

            elif action_type == "scroll":
                driver.execute_script("window.scrollBy(0, window.innerHeight * 0.8);")
                action_succeeded = True
            
            elif action_type == "extract":
                container_selector, field_xpaths = action.get("container_selector"), action.get("field_xpaths", {})
                new_items = []
                if container_selector and field_xpaths:
                    product_containers = driver.find_elements(By.XPATH, container_selector)
                    for container in product_containers:
                        item_data = {}
                        for field, xpath in field_xpaths.items():
                            try:
                                element = container.find_element(By.XPATH, f".{xpath}")
                                item_data[field] = element.text.strip() or element.get_attribute("content") or ""
                            except NoSuchElementException: item_data[field] = None
                        if item_data.get("product_title"): new_items.append(item_data)
                current_titles = {item.get('product_title') for item in state['results']}
                unique_new_items = [item for item in new_items if item.get('product_title') and item.get('product_title') not in current_titles]
                state['results'].extend(unique_new_items)
                if len(unique_new_items) > 0: state['stagnation_count'] = 0
                else: state['stagnation_count'] += 1
                push_status(job_id, "partial_result", {"new_items_found": len(unique_new_items)})
                action_succeeded = True
            
            elif action_type in ["finish_goal", "finish", "no_popup"]:
                action_succeeded = True
            
            if action_succeeded:
                if action_type != 'scroll': state['stagnation_count'] = 0
                if state['current_phase'] == 'main_task': state['sub_step'] += 1
                state['history'].append(f"Step {state['step']}: Action `{action_type}` on selector `{action.get('selector', 'N/A')}` succeeded.")
            else:
                raise Exception(f"Action '{action_type}' failed on selector '{action.get('selector')}'. Last error: {error_message}")
        except Exception as e:
            final_error_message = str(e).splitlines()[0]
            push_status(job_id, "action_failed", {"action": action, "error": final_error_message})
            state['history'].append(f"Step {state['step']}: Action `{action.get('type')}` on selector `{action.get('selector', 'N/A')}` FAILED: {final_error_message}")
            break
    state['step'] += 1
    state['history'] = state['history'][-5:]
    return state

# --- GRAPH CONTROL FLOW ---
def update_plan_node(state: AgentState) -> AgentState:
    state['plan_step'] += 1
    state['sub_step'] = 0
    push_status(state['job_id'], "plan_step_advanced", {"new_plan_step": state['plan_step']})
    return state

## --- UPGRADE --- ##
# New router to handle the outcome of the pop-up check.
def popup_router(state: AgentState) -> str:
    action_type = state['last_action'].get("type")
    if action_type == "dismiss":
        # If a pop-up needs to be dismissed, go to the execution node.
        return "execute"
    else: # This covers "no_popup"
        # If the page is clear, proceed to the main targeting logic.
        return "continue_to_targeting"

def intelligence_router(state: AgentState) -> str:
    # This router remains the same.
    if state.get('confirmed_candidate'): return "fast_path"
    else: return "slow_path"

def supervisor_router(state: AgentState) -> str:
    ## --- UPGRADE --- ##
    # This router is now phase-aware to handle the pop-up dismissal loop.
    if state['current_phase'] == "popup_check":
        # After attempting to dismiss a pop-up, always loop back to check again.
        return "recheck_popups"

    # Main task logic remains the same.
    last_action = state['last_action'].get('actions', [state['last_action']])[-1]
    last_action_type = last_action.get("type")
    if last_action_type == "finish" or state['step'] > state['max_steps']:
        return "end"
    if last_action_type == "finish_goal":
        return "update_plan"
    return "continue_goal"

def plan_completion_router(state: AgentState) -> str:
    # This router remains robust.
    if state['plan_step'] >= len(state['master_plan']):
        return "end"
    else:
        return "continue_plan"

## --- UPGRADE: GRAPH WIRING --- ##
builder = StateGraph(AgentState)
builder.add_node("navigate", navigate_to_page)
builder.add_node("planner", planner_node)
builder.add_node("dismiss_popups", dismiss_popups_node) # New node
builder.add_node("targeting", targeting_node)
builder.add_node("fast_reason", fast_reasoning_node)
builder.add_node("slow_reason", slow_reasoning_node)
builder.add_node("execute", execute_action_node)
builder.add_node("update_plan", update_plan_node)

builder.set_entry_point("navigate")
builder.add_edge("navigate", "planner")
builder.add_edge("planner", "dismiss_popups") # Planner now leads to pop-up check

# Pop-up dismissal loop
builder.add_conditional_edges(
    "dismiss_popups",
    popup_router,
    {
        "execute": "execute",
        "continue_to_targeting": "targeting"
    }
)

# Main task logic
builder.add_edge("fast_reason", "execute")
builder.add_edge("slow_reason", "execute")
builder.add_conditional_edges("targeting", intelligence_router, {"fast_path": "fast_reason", "slow_path": "slow_reason"})
builder.add_conditional_edges(
    "execute",
    supervisor_router,
    {
        "recheck_popups": "dismiss_popups", # Loop back to check for more pop-ups
        "end": END,
        "update_plan": "update_plan",
        "continue_goal": "targeting"
    }
)
builder.add_conditional_edges("update_plan", plan_completion_router, {"end": END, "continue_plan": "targeting"})
graph_app = builder.compile()

# --- JOB RUNNER AND API ---
def run_job(job_id: str, payload: dict):
    # This function's setup remains robust.
    run_dir = RUNS_DIR / job_id; screenshots_dir = run_dir / "screenshots"
    run_dir.mkdir(parents=True, exist_ok=True); screenshots_dir.mkdir(exist_ok=True)
    db_path = run_dir / "checkpoints.db"
    provider = payload["llm_provider"]
    job_analysis = { "job_id": job_id, "provider": provider, "model": MODEL_MAPPING.get(provider, "unknown"), "query": payload["query"], "url": payload["url"], "token_usage": [] }
    driver, final_state = None, {}
    try:
        options = UiAutomator2Options()
        options.platform_name = 'Android'; options.udid = "ZD222GXYPV"; options.automation_name = 'UiAutomator2'; options.browser_name = "Chrome"
        options.no_reset = True; options.auto_grant_permissions = True
        options.set_capability("appium:uiautomator2ServerInstallTimeout", 60000); options.set_capability("appium:chromedriver_autodownload", True)
        chrome_options = {"w3c": True, "args": ["--disable-fre", "--no-first-run"]}
        if payload.get("desktop_mode", False): chrome_options["args"].append("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")
        options.set_capability("goog:chromeOptions", chrome_options)
        driver = webdriver.Remote(APPIUM_SERVER_URL, options=options)
        driver.implicitly_wait(10)
        push_status(job_id, "job_started", {"provider": provider})
        objective, usage = get_objective_definition(payload["url"], payload["query"], provider)
        job_analysis["token_usage"].append({"task": "define_objective", **usage})
        push_status(job_id, "objective_defined", {"objective": objective, "usage": usage})
        initial_state = AgentState(
            job_id=job_id, driver=driver, query=payload["url"], provider=provider,
            desktop_mode=payload.get("desktop_mode", False),
            refined_query=objective.get("refined_instruction", payload["query"]),
            top_k=objective.get("top_k", 6),
            results=[], run_dir=run_dir, step=1, max_steps=40,
            last_action={}, history=[], token_usage=[],
            master_plan=[], plan_step=0, sub_step=0, stagnation_count=0,
            candidate_elements=[], confirmed_candidate=None,
            current_phase="popup_check" # Start in the popup check phase
        )
        config = {"configurable": {"thread_id": job_id}, "recursion_limit": 40}
        with SqliteSaver.from_conn_string(str(db_path)) as memory:
            graph_with_checkpointing = graph_app.with_config(checkpointer=memory)
            final_state = {}
            for chunk in graph_with_checkpointing.stream(initial_state, config=config, stream_mode="values"):
                final_state = chunk
        final_screenshots = sorted([f"runs/{job_id}/screenshots/{f.name}" for f in screenshots_dir.glob("*.png")])
        final_result = {"job_id": job_id, "objective": final_state.get('refined_query'), "top_k_goal": final_state.get('top_k'), "results": final_state.get('results', []), "screenshots": final_screenshots}
    except Exception as e:
        final_result = {"error": str(e)}; push_status(job_id, "job_failed", {"error": str(e), "trace": traceback.format_exc()})
    finally:
        JOB_RESULTS[job_id] = final_result
        push_status(job_id, "job_done")
        if 'final_state' in locals() and final_state: job_analysis["token_usage"].extend(final_state.get('token_usage', []))
        save_analysis_report(run_dir, job_analysis)
        if driver: driver.quit()

# The API endpoints remain unchanged and robust.
@app.post("/search")
async def start_search(req: SearchRequest):
    job_id = str(uuid.uuid4()); JOB_QUEUES[job_id] = asyncio.Queue()
    asyncio.get_event_loop().run_in_executor(None, run_job, job_id, req.dict())
    return {"job_id": job_id, "run_directory": f"runs/{job_id}", "stream_url": f"/stream/{job_id}", "result_url": f"/result/{job_id}"}
@app.get("/stream/{job_id}")
async def stream_status(job_id: str):
    q = JOB_QUEUES.get(job_id)
    if not q: raise HTTPException(status_code=404, detail="Job not found")
    async def event_generator():
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=60)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg["msg"] in ("job_done", "job_failed"): break
            except asyncio.TimeoutError: yield ": keep-alive\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")
@app.get("/result/{job_id}")
async def get_result(job_id: str):
    return JSONResponse(JOB_RESULTS.get(job_id, {"status": "pending"}), status_code=200 if job_id in JOB_RESULTS else 202)
@app.get("/runs/{job_id}/{folder}/{filename:path}")
async def get_run_file(job_id: str, folder: str, filename: str):
    file_path = RUNS_DIR / job_id / folder / filename
    if not file_path.is_file(): raise HTTPException(status_code=404, detail="File not found")
    if filename.endswith('.db'): return FileResponse(file_path, media_type='application/vnd.sqlite3')
    return FileResponse(file_path)
@app.get("/")
async def client_ui():
    return FileResponse(Path(__file__).parent / "static/test_client.html")