"""
VoiceGuard AI — Google Stitch Asset & Screen Fetcher
Connects to Google Stitch MCP endpoint (https://stitch.googleapis.com/mcp)
Downloads HTML, CSS tokens, and screenshots for specified project screens.
"""
import os
import sys
import json
import urllib.request
import urllib.error

STITCH_MCP_URL = "https://stitch.googleapis.com/mcp"
PROJECT_ID = "67276359575263821"

SCREENS = [
    {
        "name": "Design System",
        "id": "asset-stub-assets_c5b24c7b636248598d301b53ed776f32",
        "slug": "design_system",
        "is_asset": True
    },
    {
        "name": "VoiceGuard AI — Audio Forensics Console",
        "id": "3b339a34c9e140789523ade782d38aed",
        "slug": "audio_forensics_console",
        "is_asset": False
    },
    {
        "name": "VoiceGuard AI — Forensic History Console",
        "id": "9615459709db45149549de010b2fd146",
        "slug": "forensic_history_console",
        "is_asset": False
    },
    {
        "name": "VoiceGuard AI — Live Streaming Forensics",
        "id": "729af6a5d58441aaa75371a6a22a0d6d",
        "slug": "live_streaming_forensics",
        "is_asset": False
    },
    {
        "name": "VoiceGuard AI — Acoustic Forensics Workstation",
        "id": "80e6a271eab24cee8d8cfd80ac19e15d",
        "slug": "acoustic_forensics_workstation",
        "is_asset": False
    }
]

def mcp_call(api_key: str, tool_name: str, arguments: dict):
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "X-Goog-Api-Key": api_key.strip()
    }
    req = urllib.request.Request(STITCH_MCP_URL, data=json.dumps(payload).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            return data
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        print(f"[ERROR] HTTP {e.code} for {tool_name}: {err_body}")
        return None

def download_file(url: str, dest_path: str):
    print(f"  Downloading: {url} -> {dest_path}")
    req = urllib.request.Request(url, headers={"User-Agent": "VoiceGuard-Fetcher/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())
    print(f"  Saved ({os.path.getsize(dest_path)} bytes)")

def fetch_all(api_key: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    manifest = {}

    print(f"[*] Querying Stitch MCP for project: {PROJECT_ID}...")
    proj_res = mcp_call(api_key, "get_project", {"name": f"projects/{PROJECT_ID}"})
    if proj_res:
        with open(os.path.join(output_dir, "project_meta.json"), "w") as f:
            json.dump(proj_res, f, indent=2)
        print("[+] Project metadata saved.")

    for item in SCREENS:
        s_id = item["id"]
        slug = item["slug"]
        print(f"\n[*] Fetching Screen: {item['name']} ({s_id})...")
        
        screen_res = mcp_call(api_key, "get_screen", {
            "name": f"projects/{PROJECT_ID}/screens/{s_id}",
            "projectId": PROJECT_ID,
            "screenId": s_id
        })

        if not screen_res or "error" in screen_res:
            print(f"[-] Could not retrieve screen {s_id}")
            continue

        result_obj = screen_res.get("result", {})
        # Save screen JSON
        json_path = os.path.join(output_dir, f"{slug}.json")
        with open(json_path, "w") as f:
            json.dump(screen_res, f, indent=2)

        # Check for HTML download URL
        html_file = result_obj.get("htmlCode", {})
        if html_file.get("downloadUrl"):
            html_dest = os.path.join(output_dir, f"{slug}.html")
            try:
                download_file(html_file["downloadUrl"], html_dest)
            except Exception as ex:
                print(f"[-] Failed downloading HTML: {ex}")

        # Check for Screenshot download URL
        img_file = result_obj.get("screenshot", {})
        if img_file.get("downloadUrl"):
            img_dest = os.path.join(output_dir, f"{slug}.png")
            try:
                download_file(img_file["downloadUrl"], img_dest)
            except Exception as ex:
                print(f"[-] Failed downloading screenshot: {ex}")

        manifest[s_id] = {
            "title": item["name"],
            "slug": slug,
            "data": result_obj
        }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[DONE] All assets fetched into {output_dir}")

if __name__ == "__main__":
    key = os.environ.get("STITCH_API_KEY")
    if len(sys.argv) > 1:
        key = sys.argv[1]
    
    if not key:
        print("Usage: python scripts/fetch_stitch_assets.py <YOUR_STITCH_API_KEY>")
        print("Or set environment variable: STITCH_API_KEY")
        sys.exit(1)

    out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "stitch_assets")
    fetch_all(key, out)
