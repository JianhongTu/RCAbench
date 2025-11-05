#!/usr/bin/env python3
"""
RCAbench API Client CLI

A simple command-line client to interact with the RCAbench gateway API.

Usage:
    python client.py health
    python client.py query "Your prompt here"
    python client.py ws "Your prompt here"

Environment:
    Set RCA_HOST to the host (default: https://rcabench.nrp-nautilus.io)
    Set RCA_VERIFY_SSL to false to skip SSL verification (default: true for localhost, false for deployment)
"""

import argparse
import asyncio
import json
import os
import ssl
import sys

import requests
import websockets

# Default host
DEFAULT_HOST = "https://rcabench.nrp-nautilus.io"
RCA_HOST = os.getenv("RCA_HOST", DEFAULT_HOST)

# SSL verification: default to False for the deployment host (self-signed cert), True for localhost
DEFAULT_VERIFY_SSL = RCA_HOST == "http://localhost:8080"
RCA_VERIFY_SSL = os.getenv("RCA_VERIFY_SSL", str(DEFAULT_VERIFY_SSL)).lower() in ("true", "1", "yes")


def health_check():
    """Check the health of the service."""
    try:
        response = requests.get(f"{RCA_HOST}/health", verify=RCA_VERIFY_SSL)
        response.raise_for_status()
        print("Health check passed:", response.json())
    except requests.RequestException as e:
        print(f"Health check failed: {e}")
        sys.exit(1)


def query(prompt):
    """Send a query to the /query endpoint."""
    try:
        response = requests.post(
            f"{RCA_HOST}/query",
            json={"prompt": prompt},
            headers={"Content-Type": "application/json"},
            verify=RCA_VERIFY_SSL
        )
        response.raise_for_status()
        result = response.json()
        print("Query response:")
        print(f"Model: {result['downstream'].get('model', 'N/A')}")
        print(f"Content: {result['downstream']['content']}")
    except requests.RequestException as e:
        print(f"Query failed: {e}")
        sys.exit(1)


async def websocket_query(prompt):
    """Send a query via WebSocket."""
    # Build WebSocket URI
    if RCA_HOST.startswith("https://"):
        uri = RCA_HOST.replace("https://", "wss://") + "/ws"
    else:
        uri = RCA_HOST.replace("http://", "ws://") + "/ws"
    
    # SSL context for WebSocket
    ssl_context = None
    if uri.startswith("wss://"):
        ssl_context = ssl.create_default_context()
        if not RCA_VERIFY_SSL:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
    
    try:
        async with websockets.connect(uri, ssl=ssl_context) as ws:
            await ws.send(json.dumps({"prompt": prompt}))
            response = await ws.recv()
            result = json.loads(response)
            if "error" in result:
                print(f"WebSocket error: {result['error']}")
            else:
                print("WebSocket response:")
                print(f"Model: {result.get('model', 'N/A')}")
                print(f"Content: {result['content']}")
    except Exception as e:
        print(f"WebSocket query failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="RCAbench API Client CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Health command
    subparsers.add_parser("health", help="Check service health")

    # Query command
    query_parser = subparsers.add_parser("query", help="Send a query via HTTP POST")
    query_parser.add_argument("prompt", help="The prompt to send")

    # WebSocket command
    ws_parser = subparsers.add_parser("ws", help="Send a query via WebSocket")
    ws_parser.add_argument("prompt", help="The prompt to send")

    args = parser.parse_args()

    if args.command == "health":
        health_check()
    elif args.command == "query":
        query(args.prompt)
    elif args.command == "ws":
        asyncio.run(websocket_query(args.prompt))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
