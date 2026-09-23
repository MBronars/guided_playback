def get_adapter(name):
    if name == "paper":
        from .paper import PaperAdapter
        return PaperAdapter()
    if name == "clean_push":
        from .clean_push import CleanPushAdapter
        return CleanPushAdapter()
    raise SystemExit(f"unknown adapter {name}")
