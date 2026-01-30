# conftest.py - V8 coverage collection for Playwright tests
import json
import os
import re
import contextvars
from pathlib import Path
import pytest
import playwright.sync_api as sa

# --- configuration ---
OUT_DIR = os.environ.get("V8_COVERAGE_DIR", "coverage-raw")

# current test id (pytest nodeid)
_current_test = contextvars.ContextVar("v8cov_current_test", default="unknown")

# browsers created per test (so we can close them even on failures)
_browsers_by_test: dict[str, list] = {}

# file counter per test (if multiple browsers close in one test)
_file_counter: dict[str, int] = {}


def _sanitize(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:180] if len(s) > 180 else s


def _next_outfile(nodeid: str) -> str:
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    n = _file_counter.get(nodeid, 0) + 1
    _file_counter[nodeid] = n
    return str(Path(OUT_DIR) / f"{_sanitize(nodeid)}.{n}.precise.json")


# --- patch Playwright sync API (once, at import time) ---
_ORIG_LAUNCH = sa.BrowserType.launch
_ORIG_NEW_PAGE = sa.Browser.new_page
_ORIG_CONTEXT_NEW_PAGE = sa.BrowserContext.new_page
_ORIG_PAGE_CLOSE = sa.Page.close
_ORIG_CLOSE = sa.Browser.close


def _patched_launch(self, *args, **kwargs):
    browser = _ORIG_LAUNCH(self, *args, **kwargs)
    nodeid = _current_test.get()

    _browsers_by_test.setdefault(nodeid, []).append(browser)

    # storage for pages we started coverage on
    setattr(browser, "_v8cov_pages", [])
    setattr(browser, "_v8cov_nodeid", nodeid)

    return browser


def _arm_navigation_coverage(page, browser):
    """
    Delay starting coverage until the first real navigation and skip file:// pages.
    CDP precise coverage can hang on file:// URLs inside Docker, so we attach only
    for http/https.
    """
    try:
        if browser.browser_type.name != "chromium":
            return

        def _maybe_start_coverage(frame):
            url = frame.url or ""
            if url.startswith(("http://", "https://")):
                _start_precise_coverage(page.context, page, browser)
            # remove_listener is the public API for unsubscribing in sync API
            page.remove_listener("framenavigated", _maybe_start_coverage)

        page.on("framenavigated", _maybe_start_coverage)
    except Exception:
        # best-effort; don't break tests if event wiring fails
        pass


def _start_precise_coverage(ctx, page, browser):
    """Attach a CDP session and start precise coverage if possible."""
    try:
        url = page.url
        # Skip coverage for file:// URLs (they have security restrictions)
        if not url or url.startswith("file://"):
            return

        cdp = ctx.new_cdp_session(page)
        cdp.send("Profiler.enable")
        cdp.send("Profiler.startPreciseCoverage", {"callCount": True, "detailed": True})
        setattr(page, "_v8cov_cdp", cdp)
        setattr(page, "_v8cov_nodeid", _current_test.get())
        browser._v8cov_pages.append(page)  # type: ignore[attr-defined]
    except Exception:
        pass


def _patched_new_page(self, *args, **kwargs):
    page = _ORIG_NEW_PAGE(self, *args, **kwargs)

    try:
        # only Chromium supports CDP sessions
        _arm_navigation_coverage(page, self)
    except Exception:
        # if anything fails, don't break the test
        pass

    return page


def _patched_context_new_page(self, *args, **kwargs):
    page = _ORIG_CONTEXT_NEW_PAGE(self, *args, **kwargs)

    try:
        browser = getattr(self, "browser", None)
        if browser:
            _arm_navigation_coverage(page, browser)
    except Exception:
        pass

    return page


def _write_page_coverage(page) -> None:
    """Grab precise coverage for a single page and persist it."""
    cdp = getattr(page, "_v8cov_cdp", None)
    browser = getattr(getattr(page, "context", None), "browser", None)
    if not (cdp and browser and browser.browser_type.name == "chromium"):
        return

    # Skip file:// URLs as they have security restrictions
    try:
        url = page.url
        if url and url.startswith("file://"):
            # Clean up CDP session without trying to collect coverage
            try:
                cdp.send("Profiler.stopPreciseCoverage")
                cdp.send("Profiler.disable")
            except Exception:
                pass
            try:
                pages = getattr(browser, "_v8cov_pages", [])
                if page in pages:
                    pages.remove(page)
            except Exception:
                pass
            return
    except Exception:
        pass

    try:
        cov = cdp.send("Profiler.takePreciseCoverage")
        merged = {"result": cov.get("result", [])}
        nodeid = getattr(page, "_v8cov_nodeid", _current_test.get())
        out_file = _next_outfile(nodeid)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(merged, f)
        cdp.send("Profiler.stopPreciseCoverage")
        cdp.send("Profiler.disable")
    except Exception:
        pass
    try:
        # drop from tracking list so browser.close doesn't re-process
        pages = getattr(browser, "_v8cov_pages", [])
        if page in pages:
            pages.remove(page)
    except Exception:
        pass


def _patched_page_close(self, *args, **kwargs):
    try:
        _write_page_coverage(self)
    except Exception:
        pass
    return _ORIG_PAGE_CLOSE(self, *args, **kwargs)


def _patched_close(self, *args, **kwargs):
    # write coverage before actually closing
    try:
        if self.browser_type.name == "chromium":
            pages = getattr(self, "_v8cov_pages", [])

            for page in list(pages):
                _write_page_coverage(page)
    except Exception:
        pass
    return _ORIG_CLOSE(self, *args, **kwargs)


# apply patches
sa.BrowserType.launch = _patched_launch
sa.Browser.new_page = _patched_new_page
sa.BrowserContext.new_page = _patched_context_new_page
sa.Page.close = _patched_page_close
sa.Browser.close = _patched_close


# --- pytest glue ---
@pytest.fixture(autouse=True)
def _v8cov_test_context(request):
    token = _current_test.set(request.node.nodeid)
    try:
        yield
    finally:
        # ensure any browsers that didn't get closed (e.g., assertion failed early) are closed
        nodeid = request.node.nodeid
        for browser in _browsers_by_test.get(nodeid, []):
            try:
                if browser.is_connected():
                    browser.close()  # will trigger our patched close -> write coverage
            except Exception:
                pass

        _current_test.reset(token)
