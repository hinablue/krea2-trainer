"""Small CPU tensor cache for repeated TQD samples, with file invalidation."""

from collections import OrderedDict
import os

from safetensors.torch import load_file


_worker_cache = None
_worker_pid = None


def load_cached_tensors(path):
    """Share the 64 MiB budget across all TQD datasets in a loader process."""
    global _worker_cache, _worker_pid
    pid = os.getpid()
    if _worker_cache is None or _worker_pid != pid:
        _worker_cache, _worker_pid = TensorFileCache(), pid
    return _worker_cache.load(path)


class TensorFileCache:
    def __init__(self, max_bytes=64 * 1024 * 1024, max_entries=128):
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.size_bytes = 0
        self._entries = OrderedDict()

    def load(self, path):
        path = os.fspath(path)
        stat = os.stat(path)
        version = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        previous = self._entries.pop(path, None)
        if previous is not None:
            if previous[0] == version:
                self._entries[path] = previous
                # Callers may mutate batches; never expose cached storage.
                return {key: tensor.clone() for key, tensor in previous[1].items()}
            self.size_bytes -= previous[2]
        tensors = load_file(path)
        size = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        if self.max_entries > 0 and size <= self.max_bytes:
            while self._entries and (self.size_bytes + size > self.max_bytes or len(self._entries) >= self.max_entries):
                _, (_, _, evicted_size) = self._entries.popitem(last=False)
                self.size_bytes -= evicted_size
            # Clone away from mmap: epoch TE refresh may overwrite the same file.
            stored = {key: tensor.clone() for key, tensor in tensors.items()}
            self._entries[path] = (version, stored, size)
            self.size_bytes += size
        return tensors
