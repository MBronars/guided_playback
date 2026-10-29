"""Restart finite training loaders without hanging on an empty pass."""


def repeat_nonempty(loader):
    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("Empty training loader: check the split, exclusions and window lengths")
