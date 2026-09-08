#!/usr/bin/env python3
"""Compatibility entry point: use `python -m needle2 to-torch`."""
import sys
from needle2.cli import main
sys.argv.insert(1,'to-torch')
if __name__=='__main__': main()
