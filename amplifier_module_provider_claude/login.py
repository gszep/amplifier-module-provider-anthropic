"""Command-line login for Anthropic Claude Pro/Max OAuth."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import sys
import webbrowser

from .auth import (
    AnthropicAuthError,
    CALLBACK_HOST,
    CALLBACK_PATH,
    CALLBACK_PORT,
    REDIRECT_URI,
    authorization_url,
    default_auth_path,
    exchange_authorization_code,
    generate_pkce,
    parse_authorization_input,
    write_credentials,
)

_SUCCESS = """<!doctype html><title>Anthropic login complete</title>
<h1>Authentication complete</h1><p>You can close this window.</p>"""
_ERROR = """<!doctype html><title>Anthropic login failed</title>
<h1>Authentication failed</h1><p>{message}</p>"""


async def _read_callback(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    expected_state: str,
    result: asyncio.Future[tuple[str, str]],
) -> None:
    status = "400 Bad Request"
    body = _ERROR.format(message="Missing authorization response.")
    try:
        request_line = (await reader.readline()).decode(errors="replace").strip()
        parts = request_line.split(" ")
        if len(parts) >= 2:
            code, state = parse_authorization_input(f"http://localhost{parts[1]}")
            if not parts[1].startswith(CALLBACK_PATH):
                status = "404 Not Found"
                body = _ERROR.format(message="Callback route not found.")
            elif not code or not state:
                body = _ERROR.format(message="Missing code or state parameter.")
            elif state != expected_state:
                body = _ERROR.format(message="OAuth state mismatch.")
            else:
                status = "200 OK"
                body = _SUCCESS
                if not result.done():
                    result.set_result((code, state))
    except Exception as exc:
        body = _ERROR.format(message=str(exc))
    payload = body.encode()
    writer.write(
        f"HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\n"
        f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
        + payload
    )
    with suppress(Exception):
        await writer.drain()
    writer.close()
    with suppress(Exception):
        await writer.wait_closed()


async def login() -> None:
    verifier, challenge = generate_pkce()
    url = authorization_url(verifier, challenge)
    loop = asyncio.get_running_loop()
    callback: asyncio.Future[tuple[str, str]] = loop.create_future()

    server: asyncio.Server | None = None
    try:
        server = await asyncio.start_server(
            lambda reader, writer: _read_callback(reader, writer, verifier, callback),
            CALLBACK_HOST,
            CALLBACK_PORT,
        )
    except OSError as exc:
        print(f"Could not start the local callback server: {exc}", file=sys.stderr)

    print("Open this URL to authenticate with Anthropic:\n")
    print(url)
    print(f"\nWaiting for the callback at {REDIRECT_URI}.")
    webbrowser.open(url)

    manual = asyncio.create_task(
        asyncio.to_thread(
            input,
            "If the browser is on another machine, paste the final redirect URL or authorization code here:\n> ",
        )
    )
    callback_task = asyncio.ensure_future(callback)
    try:
        done, _ = await asyncio.wait(
            {manual, callback_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if callback_task in done:
            code, state = callback_task.result()
        else:
            code, supplied_state = parse_authorization_input(manual.result())
            state = supplied_state or verifier
            if supplied_state and supplied_state != verifier:
                raise AnthropicAuthError("OAuth state mismatch")
            if not code:
                raise AnthropicAuthError("Missing authorization code")

        credentials = await asyncio.to_thread(
            exchange_authorization_code, code, state, verifier
        )
        path = default_auth_path()
        await asyncio.to_thread(write_credentials, path, credentials)
        print(f"Anthropic OAuth credentials saved to {path}")
    finally:
        if server:
            server.close()
            await server.wait_closed()
        for task in (manual, callback_task):
            if not task.done():
                task.cancel()


def main() -> None:
    try:
        asyncio.run(login())
    except (AnthropicAuthError, KeyboardInterrupt) as exc:
        print(f"Login failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
