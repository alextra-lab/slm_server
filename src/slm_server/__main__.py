"""Main entry point for slm_server package."""

import sys

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "router":
        # Start routing service
        import uvicorn
        from slm_server.router import app

        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif len(sys.argv) > 1 and sys.argv[1] == "backends":
        # Start backend servers
        from slm_server.start_backends import main

        main()
    else:
        print("Usage: python -m slm_server [router|backends]")
        sys.exit(1)
