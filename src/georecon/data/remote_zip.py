"""Bounded HTTP-range ZIP reader: refuse a full-archive fallback."""
import io
import re
import urllib.request


class RangeZip(io.RawIOBase):
    def __init__(self, url, max_bytes=32*1024**2):
        self.url, self.max_bytes = url, max_bytes
        self.pos, self.transferred, self.ranges = 0, 0, []
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            match = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
            if response.status != 206 or not match or len(response.read(2)) != 1:
                raise ValueError("server must support exact byte ranges; full download refused")
            self.size = int(match[1])
            self.etag = response.headers.get("ETag")
        self.transferred = 1

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("invalid seek mode")
        pos = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        if pos < 0:
            raise ValueError("negative seek")
        self.pos = pos
        return pos

    def read(self, size=-1):
        size = min(self.size-self.pos, size if size >= 0 else self.size-self.pos)
        if size <= 0:
            return b""
        if self.transferred+size > self.max_bytes:
            raise ValueError("HTTP range transfer budget exceeded; full download refused")
        start, end = self.pos, self.pos+size-1
        headers = {"Range": f"bytes={start}-{end}"}
        if self.etag:
            headers["If-Match"] = self.etag
        request = urllib.request.Request(self.url, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                raise ValueError("range response mismatch; full download refused")
            if self.etag and response.headers.get("ETag") != self.etag:
                raise ValueError("archive changed during retrieval")
            data = response.read(size+1)
            if len(data) != size:
                raise ValueError("short or oversized range response")
        self.pos += size
        self.transferred += size
        self.ranges.append([start, end])
        return data
