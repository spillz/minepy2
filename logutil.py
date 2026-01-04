import os
import sys
import threading
import multiprocessing
import config

_frame_id = None
_file_handle = None
_file_lock = threading.Lock()
_file_failed = False


def _console_enabled():
    if _get_log_path():
        return False
    return True


def _get_log_path():
    loader_path = getattr(config, "LOG_LOADER_FILE_PATH", None)
    if multiprocessing.current_process().name != "MainProcess" and loader_path:
        return loader_path
    return getattr(config, "LOG_FILE_PATH", None)


def set_frame(frame_id):
    global _frame_id
    _frame_id = frame_id


def _open_log_file():
    global _file_handle
    global _file_failed
    if _file_failed:
        return None
    log_path = _get_log_path()
    if not log_path:
        return None
    if _file_handle is not None:
        return _file_handle
    with _file_lock:
        if _file_handle is not None:
            return _file_handle
        if _file_failed:
            return None
        append = bool(getattr(config, "LOG_FILE_APPEND", True))
        mode = "a" if append else "w"
        if not append and multiprocessing.current_process().name != "MainProcess":
            mode = "a"
        abs_path = os.path.abspath(log_path)
        try:
            _file_handle = open(abs_path, mode, encoding="utf-8", buffering=1)
            return _file_handle
        except OSError as exc:
            _file_failed = True
            print(
                f"LOGGING: failed to open log file '{abs_path}': {exc}. "
                "Logging disabled.",
                file=sys.stderr,
            )
            return None


def log(scope, msg, level="INFO"):
    if getattr(config, "LOG_STRICT_ONLY", False):
        if level not in ("WARN", "ERROR") and scope not in ("PIPE", "SEAM", "STRICT"):
            return
    if scope in ("FRAME", "MAINLOOP") and not getattr(config, "LOG_MAIN_LOOP", True):
        return
    pid = os.getpid()
    proc = multiprocessing.current_process().name
    thread = threading.current_thread().name
    frame = _frame_id
    frame_tag = f" f{frame}" if frame is not None else ""
    raw_text = f"[{level}{frame_tag} pid{pid} proc{proc} thr{thread} {scope}] {msg}"
    if _console_enabled():
        text = raw_text
        use_color = getattr(config, "LOG_COLOR", True) and os.getenv("NO_COLOR") is None
        if use_color:
            # Main process + main thread: default (no color).
            if proc == "MainProcess" and thread != "MainThread":
                # Main process worker thread.
                text = f"\x1b[32m{text}\x1b[0m"
            elif proc != "MainProcess":
                # Loader/external process.
                text = f"\x1b[33m{text}\x1b[0m"
        print(text)
    file_handle = _open_log_file()
    if file_handle is not None:
        with _file_lock:
            file_handle.write(raw_text + "\n")
