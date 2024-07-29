"""
Colored terminal messages using Ansi escape sequences."""

from dataclasses import dataclass


@dataclass
class SFM:
    """Colored terminal messages using Ansi escape sequences."""

    red = "\x1b[38;5;1m\x1b[1m"
    green = "\x1b[38;5;2m\x1b[1m"
    cyan = "\x1b[38;5;6m\x1b[1m"
    purple = "\x1b[38;5;5m\x1b[1m"
    yellow = "\x1b[38;5;3m\x1b[1m"
    blue = "\x1b[38;5;4m\x1b[1m"
    bold = "\x1b[1m"
    reset = "\x1b[0m"
    success = f"{green}[SUCCESS]{reset}"
    failure = f"{red}[FAILURE]{reset}"
    warning = f"{yellow}[WARNING]{reset}"
    info = f"{blue}[INFO]{reset}"
    all_succeeded = f"{green}[ALL SUCCEEDED]{reset}"
    failures_present = f"{red}[FAILURES PRESENT]{reset}"

    @classmethod
    def highlight(cls, cmd: str) -> str:
        return f"{cls.purple}{cmd}{cls.reset}"