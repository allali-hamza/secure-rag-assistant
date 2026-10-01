SEPARATOR = "-" * 30
_after_separator = False


def log_section_start(name: str, message: str) -> None:
    global _after_separator

    if not _after_separator:
        print()
        print(SEPARATOR)
        print()
    else:
        print()

    print(f"[{name}] {message}")
    _after_separator = False


def log_section_end() -> None:
    global _after_separator

    print()
    print(SEPARATOR)
    _after_separator = True


def log_step(name: str, message: str) -> None:
    print(f"[{name}] {message}")
