"""Compatibility wrapper that forwards to the production TemporalCNN trainer."""

from __future__ import annotations


def main() -> None:
    from violawake_sdk.tools.train import main as temporal_main

    temporal_main()


if __name__ == "__main__":
    main()
