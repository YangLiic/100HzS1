"""Read JPEG blob columns by byte range without caching image payloads."""
from pathlib import Path
import os

import numpy as np

from rtpolicy.datasets.lance_video import MountedFilePool


class MappedFilePool:
    """Demand-map immutable blob files and release touched PTEs after each read.

    Mapping reserves address space only. MADV_RANDOM avoids speculative reads;
    MADV_DONTNEED drops this process's residency after copying the requested
    range. Kernel filesystem caching remains managed by the OS. At most
    capacity file mappings are open, independent of dataset cardinality.
    """

    def __init__(self, capacity):
        import mmap
        from collections import OrderedDict
        if capacity < 1:
            raise ValueError('Mapped file capacity must be positive')
        if not hasattr(mmap.mmap, 'madvise'):
            raise RuntimeError('Demand-mapped JPEG reads require Linux madvise')
        self.capacity = capacity
        self._files = OrderedDict()

    def readinto(self, path, offset, buffer):
        import mmap
        if path not in self._files:
            if len(self._files) >= self.capacity:
                _, old = self._files.popitem(last=False)
                old.close()
            fd = os.open(path, os.O_RDONLY)
            try:
                mapped = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            finally:
                os.close(fd)
            mapped.madvise(mmap.MADV_RANDOM)
            self._files[path] = mapped
        self._files.move_to_end(path)
        mapped = self._files[path]
        if offset < 0:
            raise ValueError('Negative blob byte offset')
        count = min(len(buffer), max(0, len(mapped)-offset))
        if not count:
            return 0
        start = offset-offset % mmap.PAGESIZE
        end = min(len(mapped), ((offset+count+mmap.PAGESIZE-1)//mmap.PAGESIZE)*mmap.PAGESIZE)
        try:
            buffer[:count] = mapped[offset:offset+count]
        finally:
            mapped.madvise(mmap.MADV_DONTNEED, start, end-start)
        return count

    def close(self):
        for mapped in self._files.values():
            mapped.close()
        self._files.clear()


class BlobImageReader:
    """Cache blob descriptors; bound open descriptors independently of data size.

    Mounted, single-file blob-v1 fragments support native pread. Other layouts
    use Lance's public planned blob API. Neither path preloads image contents.
    """

    def __init__(self, dataset, uri, names, backend="auto", capacity=16):
        self.dataset = dataset
        self.uri = uri
        self.capacity = capacity
        self.names = set(names)
        self.pool = MappedFilePool(capacity) if backend == "mmap" else MountedFilePool(capacity)
        table = dataset.to_table(
            columns=names, blob_handling="blobs_descriptions",
            with_row_address=True, scan_in_order=True,
        )
        self.addresses = table["_rowaddr"].combine_chunks().to_numpy(zero_copy_only=False)
        self.fragments = (self.addresses >> np.uint64(32)).astype(np.uint32)
        self.metadata_bytes = table.nbytes
        self.offsets = {}
        self.sizes = {}
        self.paths = {}
        native = backend != "lance" and hasattr(os, "pread")
        for name in names:
            column = table[name].combine_chunks()
            if getattr(column.type, "names", None) != ["position", "size"] or column.null_count:
                native = False
                continue
            self.offsets[name] = column.field("position").to_numpy(zero_copy_only=False)
            self.sizes[name] = column.field("size").to_numpy(zero_copy_only=False)
        root = Path(uri).resolve()
        if native:
            for fragment in dataset.get_fragments():
                files = fragment.metadata.files
                if len(files) != 1 or getattr(files[0], "base_id", None) is not None:
                    native = False
                    break
                path = (root / "data" / files[0].path).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    native = False
                    break
                self.paths[fragment.fragment_id] = str(path)
                mask = self.fragments == fragment.fragment_id
                size = path.stat().st_size
                for name in names:
                    offsets, sizes = self.offsets[name][mask], self.sizes[name][mask]
                    if np.any(offsets > size) or np.any(sizes > size - offsets):
                        raise ValueError("Image blob range exceeds its backing file")
        if backend in ("file", "mmap") and not native:
            raise ValueError(f"image_io_backend={backend} requires mounted blob-v1 single-file fragments without external bases")
        self.backend = ("mmap" if backend == "mmap" else "file") if native else "lance"

    def take(self, rows, names):
        if self.backend == "lance":
            if self.dataset is None:
                import lance

                self.dataset = lance.dataset(self.uri)
            result = {}
            for name in names:
                values = self.dataset.read_blobs(blob_column=name, indices=rows)
                by_address = {int(address): value for address, value in values}
                result[name] = [by_address[int(self.addresses[row])] for row in rows]
            return result
        if self.pool is None:
            self.pool = MappedFilePool(self.capacity) if self.backend == "mmap" else MountedFilePool(self.capacity)
        result = {name: [None] * len(rows) for name in names}
        # Complete all camera reads for a physical file before moving on. A
        # batch spanning more files than the LRU capacity still opens each once.
        groups = {}
        for pos, row in enumerate(rows):
            groups.setdefault(int(self.fragments[row]), []).append((pos, row))
        for fragment, selected in groups.items():
            path = self.paths[fragment]
            for name in names:
                for pos, row in selected:
                    offset, size = int(self.offsets[name][row]), int(self.sizes[name][row])
                    data = bytearray(size)
                    done = 0
                    while done < size:
                        count = self.pool.readinto(path, offset + done, memoryview(data)[done:])
                        if not count:
                            raise EOFError(f"Truncated image blob at row {row}")
                        done += count
                    result[name][pos] = bytes(data)
        return result

    def close(self):
        if self.pool is not None:
            self.pool.close()

    def __getstate__(self):
        # Send only metadata to spawn workers. Native workers need neither a
        # Rust runtime nor another scan of the descriptor columns.
        state = self.__dict__.copy()
        state["dataset"] = None
        state["pool"] = None
        return state

    def __del__(self):
        if hasattr(self, "pool"):
            self.close()
