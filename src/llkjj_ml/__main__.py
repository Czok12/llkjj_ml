#!/usr/bin/env python3
"""
Entry point for python -m llkjj_ml

This allows running the ML pipeline as a module:
  python -m llkjj_ml --help
  python -m llkjj_ml process --help
  python -m llkjj_ml benchmark single --help
"""

import sys

from cli import main

if __name__ == "__main__":
    main()
    sys.exit(0)
