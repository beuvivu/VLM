"""An OS-held single-writer lease, released automatically when a process dies."""
import os
from pathlib import Path


class WriterLease:
    def __init__(self, path: Path) -> None:
        self.path, self.file = Path(path), None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('a+b')
        if self.file.seek(0, os.SEEK_END) == 0:
            self.file.write(b'0')
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            self.file = None
            raise RuntimeError('Result updater already running for this state directory') from exc
        return self

    def __exit__(self, *args):
        if self.file is not None:
            # close releases the kernel lock even after exceptions/cancellation.
            self.file.close()
            self.file = None
