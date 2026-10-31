"""Best-effort release of the page cache a full SHA-256 read leaves behind.

Hashing a multi-GB .safetensors file fills the page cache with all of it. On a
unified-memory machine (DGX Spark) those pages compete with the CUDA allocation
of the model that is loaded next. After the hash, `release_after_hash` advises
the kernel (POSIX_FADV_DONTNEED) to drop the cached pages of that one file,
through the read-only descriptor that was just hashed.

The advice never changes the file, the digest or the loaded model; a later read
loads the bytes from disk again. Unsupported platforms and refused advice are
skipped. Set MICRODUCK_KEEP_WEIGHT_CACHE=1 to keep the cache, e.g. while another
process reads the same weights.
"""
import os
import stat

# Weight files smaller than this keep their cache (processor statistics are ~20 KB,
# model weights several GB).
LARGE_BYTES = 256 * 2**20


def release_after_hash(stream):
    """Advise dropping the cached pages of a large .safetensors file just hashed.

    `stream` is the open binary reader that was hashed. Returns True only when
    the advice was given; small, non-weight, writable or non-regular files,
    missing platform support and an OSError from the kernel all return False.
    """
    advise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advise is None or dontneed is None or os.environ.get("MICRODUCK_KEEP_WEIGHT_CACHE") == "1":
        return False
    if not str(getattr(stream, "name", "")).endswith(".safetensors"):
        return False
    try:
        if not stream.readable() or stream.writable():
            return False
        fd = stream.fileno()
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size < LARGE_BYTES:
            return False
        advise(fd, 0, 0, dontneed)
    except (OSError, ValueError):
        return False
    return True
