#!/bin/sh
if [ "$(id -u)" = "0" ]; then
    chown -R app:app /app/data
    exec gosu app python run.py
else
    exec python run.py
fi
