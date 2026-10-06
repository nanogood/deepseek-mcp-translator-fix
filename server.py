#!/usr/bin/env python3
"""
DeepSeek Role Translator
=========================
HTTP 代理 + MCP 服务器。Android Studio 发来的 developer 角色消息会被自动
翻译成 DeepSeek 认识的 system 角色。

用法：
    python server.py              # stdio 模式（MCP）
    python server.py --http       # HTTP 模式（代理 + MCP）
"""

import json
import os
import traceback

import httpx
from dotenv import load_dotenv
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

# ── 加载配置 ──────────────────────────────────────────────────────────────────
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


# ── 角色翻译 ──────────────────────────────────────────────────────────────────
def translate_messages(messages: list[dict]) -> list[dict]:
    for msg in messages:
        if msg.get("role") == "developer":
            msg["role"] = "system"
    return messages


# ── 代理请求 ──────────────────────────────────────────────────────────────────
def _proxy_headers():
    return {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }


async def _proxy_chat(request: Request) -> Response:
    body = await request.body()
    data = json.loads(body)

    if "messages" in data:
        data["messages"] = translate_messages(data["messages"])
    if not data.get("model"):
        data["model"] = DEEPSEEK_MODEL

    is_stream = data.get("stream", False)

    if is_stream:
        async def streamer():
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST", f"{DEEPSEEK_BASE_URL}/chat/completions",
                    headers=_proxy_headers(), json=data,
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

        return StreamingResponse(streamer(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers=_proxy_headers(), json=data,
        )
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type="application/json")


async def _proxy_other(request: Request) -> Response:
    body = await request.body()
    path = request.url.path
    query = str(request.url.query)
    url = f"{path}?{query}" if query else path

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.request(
            method=request.method, url=f"{DEEPSEEK_BASE_URL}{url}",
            headers=_proxy_headers(), content=body or None,
        )
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=resp.headers.get("Content-Type", "application/json"))


# ── MCP 服务器 ────────────────────────────────────────────────────────────────
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

mcp = FastMCP(
    name="deepseek_mcp",
    instructions="把 developer 角色翻译成 system，透明代理 DeepSeek API",
)


class Message(BaseModel):
    role: str = Field(..., description="system / user / assistant / developer")
    content: str = Field(...)


@mcp.tool(
    name="deepseek_chat",
    annotations=ToolAnnotations(
        read_only_hint=True, destructive_hint=False,
        idempotent_hint=False, open_world_hint=True,
    ),
)
async def deepseek_chat(
    messages: list[Message],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    translated = translate_messages(
        [{"role": m.role, "content": m.content} for m in messages]
    )
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers=_proxy_headers(),
            json={
                "model": model or DEEPSEEK_MODEL,
                "messages": translated,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
    data = resp.json()
    return data["choices"][0]["message"]["content"] or ""


# ── ASGI 调度 ─────────────────────────────────────────────────────────────────
from contextlib import asynccontextmanager
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import Response

mcp_app = mcp.streamable_http_app()

# Starlette 的 Mount 不会触发子应用的 lifespan，因此由顶层应用手动启动
# FastMCP 的会话任务组（session_manager.run()）。
@asynccontextmanager
async def _lifespan(app):
    async with mcp.session_manager.run():
        yield


async def _proxy_route(request: Request) -> Response:
    path = request.url.path
    method = request.method
    try:
        if path == "/v1/chat/completions" and method == "POST":
            return await _proxy_chat(request)
        if path.startswith("/v1/"):
            return await _proxy_other(request)
    except Exception:
        traceback.print_exc()
        return Response(
            content=json.dumps({"error": traceback.format_exc()}),
            status_code=500, media_type="application/json",
        )
    return Response(
        content=json.dumps({"error": "not found"}),
        status_code=404, media_type="application/json",
    )


# 代理负责 /v1/*，其余请求（含 /mcp）交给 FastMCP 应用处理
app = Starlette(
    lifespan=_lifespan,
    routes=[
        Route("/v1/chat/completions", _proxy_route, methods=["POST"]),
        Route("/v1/{path:path}", _proxy_route),
        Mount("/", app=mcp_app),
    ],
)


# ── 入口 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="DeepSeek Role Translator")
    parser.add_argument("--http", action="store_true", help="HTTP 模式")
    parser.add_argument("--port", type=int, default=8765, help="端口（默认 8765）")
    args = parser.parse_args()

    if args.http:
        print(f"代理 + MCP 已启动: http://localhost:{args.port}")
        print(f"  代理端点:     http://localhost:{args.port}/v1")
        print(f"  MCP 端点:     http://localhost:{args.port}/mcp")
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")
    else:
        mcp.run()
