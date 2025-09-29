from __future__ import annotations
import logging, sys, warnings, threading, asyncio, traceback
from logging import Handler, FileHandler
from dataclasses import dataclass, field
from typing import Optional, Union, Callable, Any, Tuple

ExcOrStr = Union[BaseException, str]
class ErrorOutputHandler(Handler):
    def __init__(self, out_file: str, err_file: str,
        error_level: int = logging.ERROR,
        encoding: Optional[str] = "utf-8",
        delay: bool = False, duplicate_errors_to_out: bool = False,
        install_hooks: bool = True, logger_name: str = "app",
        include_asyncio: bool = True, uncaught_level: int = logging.ERROR) -> None:

        super().__init__()
        self.error_level = error_level
        self.duplicate_errors_to_out = duplicate_errors_to_out
        self.out = FileHandler(out_file, encoding=encoding, delay=delay)
        self.err = FileHandler(err_file, encoding=encoding, delay=delay)
    
        self.logger_name = logger_name
        self.include_asyncio = include_asyncio
        self.uncaught_level = uncaught_level
        # internal state
        self._installed: bool = False
        self._error_flag: bool = False
        self._fatal_flag: bool = False
        self._error_count: int = 0
        self._fatal_count: int = 0
        self._last_error_text: Optional[str] = None

        # previous hooks
        self._prev_excepthook: Optional[Callable[..., Any]] = None
        self._prev_thread_hook: Optional[Callable[..., Any]] = None
        self._prev_asyncio_handler: Optional[Callable[..., Any]] = None
        self._prev_showwarning: Optional[Callable[..., Any]] = None
        
        if install_hooks:
            self.install_hooks()
            
    # --------- logging.Handler API ----------
    def setFormatter(self, fmt: logging.Formatter) -> None:
        self.out.setFormatter(fmt)
        self.err.setFormatter(fmt)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.levelno >= self.error_level:
                # write to errors
                self.err.emit(record)
                if self.duplicate_errors_to_out:
                    self.out.emit(record)
            else:
                # write to output
                self.out.emit(record)
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        self.out.flush()
        self.err.flush()

    def close(self) -> None:
        try:
            self.out.close()
            self.err.close()
        finally:
            super().close()
    # ---------------- core helpers ----------------
    @property
    def log(self) -> logging.Logger:
        return logging.getLogger(self.logger_name)

    def _format_exc(self, e: ExcOrStr, additional: Optional[str]) -> str:
        prefix = (additional + "\n") if additional else ""
        if isinstance(e, BaseException):
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__)).rstrip()
            return prefix + tb
        return (prefix + str(e)).rstrip()

    def _record(self, text: str, fatal: bool) -> None:
        self._last_error_text = text
        self._error_flag = True
        self._error_count += 1
        if fatal:
            self._fatal_flag = True
            self._fatal_count += 1

    # ---------------- public logging API ----------------
    def output(self, msg: str) -> None:
        self.log.info(msg)

    def error(self, e: ExcOrStr, *, additional: Optional[str] = None, fatal: bool = False) -> None:
        text = self._format_exc(e, additional)
        self._record(text, fatal=fatal)
        # log with traceback iff we got an Exception
        if isinstance(e, BaseException):
            self.log.error(text, exc_info=e)
        else:
            self.log.error(text)

    # ---------------- hooks: capture everything, never exit ----------------
    def install_hooks(self) -> None:
        if self._installed:
            return
        self._installed = True

        # warnings -> logging (as ERROR to ensure visibility)
        self._prev_showwarning = warnings.showwarning
        logging.captureWarnings(True)
        warnings.filterwarnings("default")

        # main thread uncaught
        self._prev_excepthook = sys.excepthook
        sys.excepthook = self._excepthook

        # other threads (py3.8+)
        if hasattr(threading, "excepthook"):
            self._prev_thread_hook = threading.excepthook
            threading.excepthook = self._threading_excepthook  # type: ignore[attr-defined]

        # asyncio unhandled task exceptions
        if self.include_asyncio:
            try:
                loop = asyncio.get_event_loop()
                self._prev_asyncio_handler = loop.get_exception_handler()
                loop.set_exception_handler(self._asyncio_handler)
            except RuntimeError:
                pass  # no loop yet; safe to ignore

    def uninstall_hooks(self) -> None:
        if not self._installed:
            return
        self._installed = False

        logging.captureWarnings(False)
        if self._prev_showwarning:
            warnings.showwarning = self._prev_showwarning  # type: ignore[assignment]
        if self._prev_excepthook:
            sys.excepthook = self._prev_excepthook
        if self._prev_thread_hook and hasattr(threading, "excepthook"):
            threading.excepthook = self._prev_thread_hook  # type: ignore[assignment]
        if self._prev_asyncio_handler is not None:
            try:
                loop = asyncio.get_event_loop()
                loop.set_exception_handler(self._prev_asyncio_handler)
            except RuntimeError:
                pass

    # ---------------- checkpoints ----------------
    def checkpoint(self, *, exit_on_error: bool = True, fatal_only: bool = False) -> bool:
        """
        Returns True if errors have occurred (or only fatal if fatal_only=True).
        Optionally sys.exit(1). Never clears flags.
        """
        condition = self._fatal_flag if fatal_only else self._error_flag
        if condition:
            note = "fatal" if fatal_only else "any"
            preview = (self._last_error_text[:200] + "…") if (self._last_error_text and len(self._last_error_text) > 200) else self._last_error_text
            self.log.error("Checkpoint: %s errors detected. last_error=%r", note, preview)
            if exit_on_error:
                sys.exit(1)
            return True
        return False

    # utilities
    def has_errors(self) -> bool: return self._error_flag
    def has_fatal(self) -> bool:  return self._fatal_flag
    def counts(self) -> Tuple[int, int]: return self._error_count, self._fatal_count
    def clear_errors(self) -> None:
        self._error_flag = self._fatal_flag = False
        self._error_count = self._fatal_count = 0
        self._last_error_text = None

    # ---------------- internal hook handlers ----------------
    def _excepthook(self, exc_type, exc_value, exc_tb):
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb)).rstrip()
        self._record(text, fatal=True)  # treat uncaught as fatal
        self.log.log(self.uncaught_level, "Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        if self._prev_excepthook:
            self._prev_excepthook(exc_type, exc_value, exc_tb)

    def _threading_excepthook(self, args):  # threading.ExceptHookArgs
        header = f"Uncaught thread exception in {getattr(args.thread, 'name', 'Thread')}"
        text = header + "\n" + "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)).rstrip()
        self._record(text, fatal=True)
        self.log.log(self.uncaught_level, header, exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        if self._prev_thread_hook:
            self._prev_thread_hook(args)

    def _asyncio_handler(self, loop, context):
        msg = context.get("message", "Unhandled asyncio exception")
        exc = context.get("exception")
        if exc is not None:
            self._record(f"{msg}", fatal=True)
            self.log.log(self.uncaught_level, msg, exc_info=exc)
        else:
            text = str(context)
            self._record(text, fatal=True)
            self.log.log(self.uncaught_level, text)


