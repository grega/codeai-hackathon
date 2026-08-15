"""Generate an animation from a text prompt and preview it on the rigged model.

Generated .glb files are written to output/ (created if needed) alongside this
script, regardless of the current working directory.

Usage:
    python main.py "wave arms in air"
    python main.py "spin around in a circle" --input rigged_human.glb --output spin.glb
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.parse

import webview
from dotenv import load_dotenv

import animator
import gltf_utils
import llm_client
import server

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "anthropic/claude-sonnet-5"


def build(prompt: str, input_path: str, output_path: str, use_llm: bool = True) -> str:
    gltf, joints = gltf_utils.load_skeleton(input_path)
    name, tracks = animator.generate(prompt, gltf, joints, use_llm=use_llm)
    gltf_utils.add_rotation_animation(gltf, tracks, name=name)
    gltf_utils.save(gltf, output_path)
    return name


def _status_url(base_url: str, *, prompt: str | None = None, error: str | None = None) -> str:
    # A served file, not a data: URL -- pywebview's macOS backend doesn't pass data:
    # URLs to the native webview as-is, it routes them through its own local server
    # and 404s. Serving a real file sidesteps that entirely.
    params = {"error": error} if error is not None else {"prompt": prompt or ""}
    return f"{base_url}/status.html?{urllib.parse.urlencode(params)}"


def main() -> None:
    load_dotenv()  # picks up OPENROUTER_API_KEY from ../.env (searches upward from this file)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help='e.g. "wave arms in air" or "spin around in a circle"')
    parser.add_argument("--input", default="rigged_human.glb")
    parser.add_argument("--output", default="out.glb", help="filename written inside output/")
    parser.add_argument("--no-llm", action="store_true", help="skip the LLM, use hardcoded wave/spin only")
    parser.add_argument(
        "--openrouter", action="store_true",
        help=f"use {OPENROUTER_MODEL} via OpenRouter instead of a local MLX server (needs OPENROUTER_API_KEY)",
    )
    parser.add_argument("--model", default=None, help="model name as the server/provider expects it")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint URL")
    args = parser.parse_args()

    if args.openrouter:
        llm_client.DEFAULT_MODEL = args.model or OPENROUTER_MODEL
        llm_client.DEFAULT_BASE_URL = args.base_url or OPENROUTER_BASE_URL
        if not os.environ.get("OPENROUTER_API_KEY"):
            sys.exit("Error: --openrouter needs OPENROUTER_API_KEY set (checked .env and the environment)")
    else:
        llm_client.DEFAULT_MODEL = args.model or llm_client.DEFAULT_MODEL
        llm_client.DEFAULT_BASE_URL = args.base_url or llm_client.DEFAULT_BASE_URL

    directory = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(directory, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.basename(args.output)  # strip any directory the caller passed -- always lands in output/
    output_path = os.path.join(output_dir, output_filename)

    base_url = server.serve_directory(directory)  # started up front, serves both status.html and viewer.html

    window = webview.create_window(
        f"Animation preview: {args.prompt}",
        _status_url(base_url, prompt=args.prompt),
        width=900, height=700,
    )

    def _generate_then_load(window: webview.Window) -> None:
        try:
            anim_name = build(args.prompt, args.input, output_path, use_llm=not args.no_llm)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            window.load_url(_status_url(base_url, error=str(e)))
            return
        url = f"{base_url}/viewer.html?src=output/{output_filename}&anim={anim_name}"
        window.load_url(url)

    webview.start(_generate_then_load, window)


if __name__ == "__main__":
    main()
