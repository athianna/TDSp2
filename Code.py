from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64, json, re, io, os, traceback
from pathlib import Path
import google.generativeai as genai

# ========================
# CONFIGURATION
# ========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <-- for production, set specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAFE_LIBS = [
    "pandas", "matplotlib.pyplot", "numpy", "requests",
    "bs4", "re", "io", "json", "base64", "datetime", "scipy.stats"
]
MAX_RETRIES = 3

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ========================
# FILE LOADING
# ========================
def load_file_content(filepath: Path) -> Union[str, bytes, pd.DataFrame]:
    """Load a file from local disk into memory in the right format."""
    ext = filepath.suffix.lower()
    try:
        if ext in [".csv"]:
            return pd.read_csv(filepath)
        elif ext in [".xls", ".xlsx"]:
            return pd.read_excel(filepath)
        elif ext in [".json"]:
            return json.load(open(filepath, "r", encoding="utf-8"))
        elif ext in [".html", ".htm"]:
            return open(filepath, "r", encoding="utf-8").read()
        elif ext in [".txt", ".sql", ".md"]:
            return open(filepath, "r", encoding="utf-8").read()
        elif ext in [".png", ".jpg", ".jpeg", ".gif"]:
            return filepath.read_bytes()
        else:
            return filepath.read_bytes()  # Fallback: binary
    except Exception as e:
        return {"error": f"Could not load {filepath.name}: {str(e)}"}

async def preprocess_files(uploaded: Dict[str, UploadFile]) -> Dict[str, Union[str, bytes, pd.DataFrame]]:
    """Process uploaded + local files into memory objects."""
    processed = {}

    # First load uploaded files
    for name, file in uploaded.items():
        content = await file.read()
        ext = Path(name).suffix.lower()

        try:
            if ext == ".csv":
                def safe_encode(obj):
                        if isinstance(obj, pd.DataFrame):
                            obj = obj.to_csv(index=False).encode("utf-8")
                        if isinstance(obj, str):
                            obj = obj.encode("utf-8")
                        if isinstance(obj, bytes):
                            return base64.b64encode(obj).decode("utf-8")
                        return obj
                processed[name] = pd.read_csv(io.BytesIO(content))
            elif ext in [".xls", ".xlsx"]:
                processed[name] = pd.read_excel(io.BytesIO(content))
            elif ext in [".json"]:
                processed[name] = json.loads(content.decode("utf-8"))
            elif ext in [".txt", ".sql", ".html", ".htm", ".md"]:
                processed[name] = content.decode("utf-8")
            else:
                processed[name] = content  # binary
        except Exception as e:
            processed[name] = {"error": f"Failed to read {name}: {str(e)}"}

    return processed

# ========================
# TASK BREAKDOWN & CODE GEN
# ========================
def get_task_breakdown(question: str, file_names: List[str]) -> str:
    prompt = (
    "You are a helpful and expert data analyst assistant AI.\n\n"
    "The user has uploaded a file named `questions.txt` containing the main task description:\n"
    f"{question}\n\n"
    f"The following additional files have also been uploaded: {', '.join(file_names)}\n\n"
    
    "Your job is to break down the task into a clear and logical sequence of data analysis steps.\n"
    "Do NOT write code. Do NOT read from or write to local disk files.\n\n"
    "Instead, write a high-level plan of steps that should be followed to complete the task, including:\n"
    "- Web scraping (if any) from specific links or ```sql .....``` or files as mentioned\n"
    "- Reading data , column names and data type from the uploaded files carefully and correctly\n"
    "- Use the data properly to answer each questions seperately\n"
    "- Returning the final results in a JSON response format\n\n"

    "When referring to uploaded files or URLs, mention them exactly as provided.\n"
    "The final step must include instructions to return a structured JSON response and the format of answers to each question as described in `questions.txt`."
    )
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

def generate_code(steps: str) -> str:
    prompt = (
"You are a skilled Python coding assistant.\n\n" 
"Input:\n" 
"- Understand the Task breakdown of analysis step by step and scrape the data from the right source (Website or local files).\n" 
"- Answer each question separately and appropriately using the cleaned data.\n"

"Rules:\n" 
"1. Use files directly — no re-reading from disk.\n"
"2. DataFrames are not always ready-to-use , But I have preprocessed them. So understand the structure of the data and extract properly\n"
"3. For DataFrame → text: .to_csv(index=False); for bytes: .to_csv(index=False).encode('utf-8').\n" 
"4. Never .encode() a DataFrame or pass it directly to io.StringIO/io.BytesIO — convert first.\n" 
"5. Always check isinstance() before processing.\n" 
"6. Detect column names dynamically — do NOT hardcode unless explicitly stated in the task breakdown.\n" 
" - If expected columns are missing, search for similar names (case-insensitive, partial match).\n" 
" - If still unavailable, skip that step and add a note in the result instead of failing.\n" 
"7. Return JSON-serializable dict only; include { 'error': '<msg>' } if needed.\n\n" 

f"--- Task Breakdown ---\n{steps}\n\n" 
"""Write a single function:\n 
```python\n
    def analyze(files: dict) -> dict:\n
       # analysis here\n
       return { 'result': ... }\n
```
    """
"Ensure robustness to unexpected formats, adapt to available column names, and produce all answers per the task breakdown." 
)


    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    code = re.sub(r"^```(?:python)?", "", model.generate_content(prompt).text.strip(), flags=re.IGNORECASE).strip("`")
    return code

def fix_code_with_gemini(code: str, error: str) -> str:
    fix_code_prompt = (
    f"The following Python code fails with an error:\n\n"
    f"--- Code ---\n{code}\n\n"
    f"--- Error ---\n{error}\n\n"
    "Rewrite the entire Python function in a corrected and generalized way so that this error and other similar errors "
    "do not occur again.\n\n"
    "Requirements:\n"
    "- Keep the function signature exactly as `def analyze(files: dict) -> dict:`.\n"
    "- For each file in `files`, robustly handle different input types:\n"
    "    * If file content is bytes, decode it safely.\n"
    "    * If file content is string (raw or base64), parse/convert appropriately.\n"
    "    * If content is already a DataFrame-like object, accept it.\n"
    "    * If unsupported, record a structured error instead of raising.\n"
    "- Auto-detect format (CSV, Excel, JSON, Parquet) and try to read; never reject without attempting.\n"
    "- Always catch and log file-specific errors but continue processing others.\n"
    "- Ensure the code runs end-to-end without repeating the same error.\n"
    "- Return results in a dictionary with either computed answers or structured error messages per file.\n"
    "- Use only these libraries: {', '.join(SAFE_LIBS)}.\n"
    "- Output only the fully corrected Python code (no explanations, no markdown)."
)



    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    response = model.generate_content(fix_code_prompt)
    fixed = re.sub(r"^```(?:python)?", "", response.text.strip(), flags=re.IGNORECASE).strip("`")
    return fixed

# ========================
# CODE EXECUTION
# ========================
def execute_code(code: str, files: dict) -> dict:
    
    def clean_code(code: str) -> str:
        return re.sub(r"^```(?:python)?", "", code.strip(), flags=re.IGNORECASE).strip("`")

    def has_nested_error(result: dict) -> bool:
        if not isinstance(result, dict):
            return False
        if "error" in result:
            return True
        if "result" in result and isinstance(result["result"], dict):
            return has_nested_error(result["result"])
        return False
    code = clean_code(code)

    for attempt in range(MAX_RETRIES):
        try:
            exec_globals = {}
            exec(code, exec_globals)

            analyze_func = exec_globals.get("analyze")
            if not analyze_func:
                raise Exception("Function 'analyze' not defined.")

            result = analyze_func(files)
            if is_error_result(result) or has_nested_error(result):
                raise Exception(f"Detected error in result: {result}")

            return {"success": True, "result": result, "attempt": attempt + 1}

        except Exception as e:
            traceback_str = traceback.format_exc()

            if attempt < MAX_RETRIES - 1:
                code = fix_code_with_gemini(code, traceback_str)
            else:
                return {
                    "success": False,
                    "error": traceback_str,
                    "attempts": attempt + 1,
                    "last_code": code
                }

    return {"success": False, "error": "Max retries exceeded."}

def make_json_safe(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    return obj

def is_error_result(result):
    if isinstance(result, str):
        return any(word in result.lower() for word in ["error", "exception", "missing", "invalid"])
    if isinstance(result, dict):
        return any(is_error_result(v) for v in result.values())
    if isinstance(result, list):
        return any(is_error_result(v) for v in result)
    return False

# ========================
# FASTAPI ENDPOINT
# ========================
@app.post("/api")
async def process_files(files: List[UploadFile] = File(...)):
    try:
        uploaded_files = {file.filename: file for file in files}
        if "questions.txt" not in uploaded_files:
            return JSONResponse(status_code=400, content={"error": "'questions.txt' is required."})

        # Preprocess all files (uploaded + local)
        processed_files = await preprocess_files(uploaded_files)

        # Extract question text
        question_text = processed_files["questions.txt"] if isinstance(processed_files["questions.txt"], str) else ""
        file_names = list(processed_files.keys())

        # Generate plan & code
        breakdown = get_task_breakdown(question_text, file_names)
        code = generate_code(breakdown)
        # Execute code
        result = execute_code(code, processed_files)
        safe_result = make_json_safe(result)
        if safe_result["success"]:
            return JSONResponse(content={"result": safe_result["result"]})
        else:
            return JSONResponse(status_code=500, content={
                "error": safe_result.get("error", "Unknown error"),
            })


    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
