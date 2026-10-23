#!/usr/bin/env python3
"""Preview the static course locally, including seekable video playback."""

import argparse
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re


class CourseHandler(SimpleHTTPRequestHandler):
    """Serve single byte ranges used by browser media players."""

    def send_head(self):
        self.remaining = None
        path = Path(self.translate_path(self.path))
        requested = self.headers.get("Range")
        # Delegate directories, conditional requests and ordinary assets to Python.
        if not requested or not path.is_file() or self.headers.get("If-Range"):
            return super().send_head()

        match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip())
        if not match or not any(match.groups()):
            return super().send_head()

        try:
            stream = path.open("rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return None

        size = path.stat().st_size
        first, last = match.groups()
        if first:
            start = int(first)
            end = min(int(last), size - 1) if last else size - 1
        else:
            start = max(0, size - int(last))
            end = size - 1

        if start > end or start >= size:
            stream.close()
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None

        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        stream.seek(start)
        self.remaining = end - start + 1
        return stream

    def copyfile(self, source, outputfile):
        if self.remaining is None:
            return super().copyfile(source, outputfile)
        # Do not send bytes beyond the range, even when the file is much larger.
        remaining = self.remaining
        while remaining:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args()
    web = Path(__file__).resolve().parents[1] / "web"
    handler = partial(CourseHandler, directory=str(web))
    with ThreadingHTTPServer((args.bind, args.port), handler) as server:
        print(f"Course: http://{args.bind}:{args.port}/zh/index.html", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
