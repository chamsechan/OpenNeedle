#!/usr/bin/env python3
"""Compatibility entry point: use `python -m needle2 quantize`."""
import sys
from needle2.cli import main
sys.argv.insert(1,'quantize')
if __name__=='__main__': main()
