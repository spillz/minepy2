import os
import threading
import multiprocessing
import config

_frame_id = None


def set_frame(frame_id):
    global _frame_id
    _frame_id = frame_id


def log(scope, msg, level="INFO"):
    if scope in ("FRAME", "MAINLOOP") and not getattr(config, "LOG_MAIN_LOOP", True):
        return
    pid = os.getpid()
    proc = multiprocessing.current_process().name
    thread = threading.current_thread().name
    frame = _frame_id
    frame_tag = f" f{frame}" if frame is not None else ""
    text = f"[{level}{frame_tag} pid{pid} proc{proc} thr{thread} {scope}] {msg}"
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
