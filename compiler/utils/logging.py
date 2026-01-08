"""
Unified logging for the compiler.

Provides dual output to console and build.log file.
Tracks warnings and errors for end-of-build summary.

Usage:
    from utils import log, logWarning, logError, logDebug, init_logging, print_summary

    # At start of main script:
    init_logging()

    # Throughout code:
    log("Building cross tables...")           # Info - section headers, major points
    logWarning("file not found")              # May cause issues with output
    logError("critical failure")              # Fundamentally breaks output
    logDebug("processing vertex 123")         # Useful for debugging

    # At end:
    print_summary()  # Shows warning/error counts
"""

import sys
import atexit
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

# ANSI color codes
class Colors:
    YELLOW = '\033[93m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


# Module state
_log_file = None
_log_path = None
_initialized = False
_warnings: List[str] = []
_errors: List[str] = []


def init_logging(log_path: Path = None):
    """
    Initialize logging to both console and file.

    Args:
        log_path: Path to log file. Defaults to ../build.log (project root)
    """
    global _log_file, _log_path, _initialized, _warnings, _errors

    if _initialized:
        return

    # Reset tracking lists
    _warnings = []
    _errors = []

    if log_path is None:
        # Default to project root (one level up from compiler/)
        log_path = Path(__file__).parent.parent.parent / "build.log"

    _log_path = Path(log_path)
    _log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _log_file = open(_log_path, 'w', encoding='utf-8')
        _initialized = True

        # Write header
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        _log_file.write(f"Build started: {timestamp}\n")
        _log_file.write("=" * 70 + "\n\n")
        _log_file.flush()

        # Register cleanup
        atexit.register(close_logging)

    except Exception as e:
        print(f"Warning: Could not open log file {_log_path}: {e}", file=sys.stderr)
        _log_file = None


def close_logging():
    """Close the log file."""
    global _log_file, _initialized

    if _log_file is not None:
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            _log_file.write(f"\n{'=' * 70}\n")
            _log_file.write(f"Build finished: {timestamp}\n")
            _log_file.close()
        except Exception:
            pass
        _log_file = None

    _initialized = False


def print_summary():
    """
    Print a summary of warnings and errors at the end of the build.
    Uses colors for terminal output.
    """
    global _warnings, _errors

    log("\n" + "=" * 70)
    log("BUILD SUMMARY")
    log("=" * 70)

    # Print error details first
    if _errors:
        print(f"\n{Colors.RED}{Colors.BOLD}Errors ({len(_errors)}):{Colors.RESET}")
        for err in _errors:
            print(f"  {Colors.RED}- {err}{Colors.RESET}")
        # Also write to log file (without colors)
        if _log_file:
            _log_file.write(f"\nErrors ({len(_errors)}):\n")
            for err in _errors:
                _log_file.write(f"  - {err}\n")

    # Print warning details
    if _warnings:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}Warnings ({len(_warnings)}):{Colors.RESET}")
        for warn in _warnings:
            print(f"  {Colors.YELLOW}- {warn}{Colors.RESET}")
        # Also write to log file (without colors)
        if _log_file:
            _log_file.write(f"\nWarnings ({len(_warnings)}):\n")
            for warn in _warnings:
                _log_file.write(f"  - {warn}\n")

    # Print final counts
    print()
    if _errors:
        print(f"{Colors.RED}{Colors.BOLD}{len(_errors)} Error(s){Colors.RESET}", end="")
    else:
        print(f"{Colors.GREEN}0 Errors{Colors.RESET}", end="")

    print(" | ", end="")

    if _warnings:
        print(f"{Colors.YELLOW}{Colors.BOLD}{len(_warnings)} Warning(s){Colors.RESET}")
    else:
        print(f"{Colors.GREEN}0 Warnings{Colors.RESET}")

    # Write counts to log file
    if _log_file:
        _log_file.write(f"\n{len(_errors)} Error(s) | {len(_warnings)} Warning(s)\n")
        _log_file.flush()


def get_counts() -> Tuple[int, int]:
    """Return (error_count, warning_count)."""
    return len(_errors), len(_warnings)


def _write_to_file(msg: str, end: str = "\n"):
    """Write message to log file."""
    if _log_file is not None:
        try:
            _log_file.write(msg + end)
            _log_file.flush()
        except Exception:
            pass


def log(msg: str = "", end: str = "\n"):
    """
    Log an info message to both console and file.
    Use for section headers and major points in the build process.
    """
    if not _initialized:
        init_logging()

    print(msg, end=end)
    _write_to_file(msg, end)


def logWarning(msg: str, end: str = "\n"):
    """
    Log a warning message. Warnings indicate something may cause issues with output.
    Displayed in yellow. Tracked for end-of-build summary.
    """
    global _warnings

    if not _initialized:
        init_logging()

    formatted = f"Warning: {msg}"
    print(f"{Colors.YELLOW}{formatted}{Colors.RESET}", end=end)
    _write_to_file(formatted, end)
    _warnings.append(msg)


def logError(msg: str, end: str = "\n"):
    """
    Log an error message. Errors indicate something fundamentally breaks the output.
    Displayed in red. Tracked for end-of-build summary.
    """
    global _errors

    if not _initialized:
        init_logging()

    formatted = f"ERROR: {msg}"
    print(f"{Colors.RED}{formatted}{Colors.RESET}", end=end, file=sys.stderr)
    _write_to_file(formatted, end)
    _errors.append(msg)


def logDebug(msg: str, end: str = "\n"):
    """
    Log a debug message. Only written to log file, not shown in console.
    Use for detailed information useful when debugging issues.
    """
    if not _initialized:
        init_logging()

    formatted = f"[DEBUG] {msg}"
    _write_to_file(formatted, end)
