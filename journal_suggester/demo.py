"""CPU-only interface demonstration with clearly labelled fictional records."""
def serve_demo(port=8765):
    from .app import make_server
    from .resources import ROOT
    from .web_service import SearchService
    service = SearchService(ROOT / "demo")
    server = make_server(service, port)
    print(f"Synthetic lexical demo: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
