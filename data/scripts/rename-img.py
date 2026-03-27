from pathlib import Path


def rename_images(folder: str | Path = ".", prefix: str = "test", start: int = 0):
    folder = Path(folder)
    extens = {".png", ".jpg", ".jpeg"}

    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extens]
    files.sort(key=lambda p: p.name.lower())

    i = start
    for p in files:
        new_path = folder / f"{prefix}{i}{p.suffix.lower()}"
        while new_path.exists() and new_path != p:
            i += 1
            new_path = folder / f"{prefix}{i}{p.suffix.lower()}"

        if new_path != p:
            p.rename(new_path)
        i += 1


if __name__ == "__main__":
    rename_images(folder="./img/", prefix="test", start=0)
