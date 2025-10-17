#///script
#requires-python = ">=3.11"
#dependecies = ["requests", "python-dotenv"]
#///script

import os
import re
import json
import time
import base64
import requests
import uuid
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, status
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

# ----------------------------
# 🔧 Configuration
# ----------------------------
load_dotenv()
API_SECRET = os.getenv("SECRET")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([GITHUB_TOKEN, GITHUB_USERNAME, GEMINI_API_KEY]):
    raise RuntimeError("❌ Missing environment variables")

genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="FastAPI + Gemini GitHub Automation")

# ----------------------------
# 📦 Request Model
# ----------------------------
# class TaskRequest(BaseModel):
#     email: str
#     secret: str
#     task: str
#     brief: str | None = None
#     evaluation_url: str | None = None

# ----------------------------
# 🔐 Secret Validation
# ----------------------------
def validate_secret(secret: str):
    if secret != API_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid secret key",
        )

# ----------------------------
# 🤖 Gemini Content Generator
# ----------------------------
def generate_project_code(task: str, brief: str | None):
    """Generate code files using Gemini API."""
    prompt = f"""
    Generate a web project for "{task}". Additional info: {brief if brief else 'N/A'}.
    Return a JSON object with 5 keys: index.html, style.css, script.js, LICENSE, README.md.
    Each key's value is the full text of that file.
    Output *only* valid JSON, no markdown or commentary.
    """    

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        files = json.loads(resp.text)
    except Exception:
        print("❌ Gemini API error or invalid JSON response")
        files = {
            "index.html": "<!DOCTYPE html><html><head><link rel='stylesheet' href='style.css'></head><body><h1>Captcha Solver</h1><script src='script.js'></script></body></html>",
            "style.css": "body { font-family: sans-serif; text-align: center; margin-top: 50px; }",
            "script.js": "// TODO: Implement captcha fetching and solving",
            "LICENSE": "MIT License",
            "README.md": f"# {task}\n\nAI-generated project."
        }
    return files


def parse_ai_text(ai_text: str) -> dict:
    files = {}
    pattern = r"---\s*(.+?)\s*---\s*([\s\S]*?)(?=(?:---\s*.+?\s*---)|$)"
    matches = re.findall(pattern, ai_text)
    for filename, content in matches:
        files[filename.strip()] = content.strip()
    # Ensure all required files exist
    for f in ["index.html","style.css","script.js","LICENSE","README.md"]:
        if f not in files:
            files[f] = ""
    return files

# ----------------------------
# 🧩 GitHub Helpers
# ----------------------------
def github_api(url: str, method: str = "GET", data: dict | None = None):
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    if method == "POST":
        return requests.post(url, headers=headers, json=data)
    elif method == "PUT":
        return requests.put(url, headers=headers, json=data)
    elif method == "PATCH":
        return requests.patch(url, headers=headers, json=data)
    else:
        return requests.get(url, headers=headers)

def create_github_repo(repo_name: str):
    url = "https://api.github.com/user/repos"
    data = {"name": repo_name, "private": False, "auto_init": True}
    r = github_api(url, "POST", data)
    r.raise_for_status()
    repo = r.json()

    # Wait until GitHub registers the repo
    for _ in range(5):
        check = github_api(repo["url"])
        if check.status_code == 200:
            break
        time.sleep(2)
    return repo["html_url"]

def commit_files(repo_name: str, files: dict):
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

    def create_blob(path, content):
        for attempt in range(3):
            r_blob = requests.post(
                f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/git/blobs",
                headers=headers,
                json={"content": content, "encoding": "utf-8"},
            )
            if r_blob.status_code == 201:
                return r_blob.json()["sha"]
            elif r_blob.status_code == 409:
                print(f"⚠️ Blob conflict on {path}, retrying...")
                time.sleep(2)
            else:
                r_blob.raise_for_status()
        raise RuntimeError(f"Failed to create blob for {path}")

    blobs = {path: create_blob(path, content) for path, content in files.items()}

    tree_items = [
        {"path": path, "mode": "100644", "type": "blob", "sha": sha}
        for path, sha in blobs.items()
    ]

    r_tree = requests.post(
        f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/git/trees",
        headers=headers,
        json={"tree": tree_items},
    )
    r_tree.raise_for_status()
    tree_sha = r_tree.json()["sha"]

    r_commit = requests.post(
        f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/git/commits",
        headers=headers,
        json={"message": "Initial commit", "tree": tree_sha},
    )
    r_commit.raise_for_status()
    commit_sha = r_commit.json()["sha"]

    # 3️⃣ Update main branch (create if missing)
    ref_url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/git/refs/heads/main"
    r_ref = requests.patch(ref_url, headers=headers, json={"sha": commit_sha, "force": True})

    # If the branch doesn't exist (e.g., auto_init=False case), create it
    if r_ref.status_code == 404:
        r_ref = requests.post(
            f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/git/refs",
            headers=headers,
            json={"ref": "refs/heads/main", "sha": commit_sha},
        )

    r_ref.raise_for_status()


    return commit_sha




def enable_github_pages(repo_name: str):
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/pages"
    data = {"source": {"branch": "main", "path": "/"}}
    r = github_api(url, "POST", data)
    if r.status_code not in (200, 201):
        print("⚠️ GitHub Pages enable warning:", r.text)
    return f"https://{GITHUB_USERNAME}.github.io/{repo_name}/"

# ----------------------------
# 🧠 Background Job
# ----------------------------
def background_job(data: dict):
    try:
        task = data["task"]
        print("Starting background job for:", task)
        files = generate_project_code(data.get("task"), data.get("brief"))
        # Unique repo name
        repo_name = f"{task.replace(' ', '-')}-{uuid.uuid4().hex[:8]}"
        repo_url = create_github_repo(repo_name)
        commit_sha = commit_files(repo_name, files)
        pages_url = enable_github_pages(repo_name)

        # Optional: wait for GitHub Pages to be live
        time.sleep(40)

        # Send final JSON to callback
        if data.get("evaluation_url"):
            requests.post(data["evaluation_url"], json={
                "email": data.get("email"),
                "task": data.get("task"),
                "round": data.get("round"),
                "nonce": data.get("nonce"),
                "repo_url": repo_url,
                "pages_url": pages_url,
                "commit_sha": commit_sha,
            })
        print("✅ Background job finished:", pages_url)
    except Exception as e:
        print("❌ Background job failed:", e)

# ----------------------------
# 🚀 FastAPI Endpoint
# ----------------------------
@app.post("/handle_request/")
async def handle_request(req: dict, background_tasks: BackgroundTasks):
    validate_secret(req.get("secret", ""))
    background_tasks.add_task(background_job, req)
    # Respond immediately
    return {"status": "accepted", "message": "Task received"}

# ----------------------------
# ▶️ Run server
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860) #port=7860 for huggingface spaces compatibility