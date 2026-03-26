#!/usr/bin/env python3
"""Run the ViolaWake Console backend."""
import os

import uvicorn

if __name__ == "__main__":
    env = os.environ.get("VIOLAWAKE_ENV", "development")
    is_dev = env != "production"

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("VIOLAWAKE_PORT", "8000")),
        reload=is_dev,
    )
