from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import threading
import unittest
import zipfile
from georecon.data.remote_zip import RangeZip


@contextmanager
def server(payload, ranges=True, wrong_etag=False, wrong_range=False):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
            body = payload[start:end+1]
            self.send_response(206 if ranges else 200)
            self.send_header("Content-Range", f"bytes {start+int(wrong_range)}-{end}/{len(payload)}")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("ETag", '"changed"' if wrong_etag and start != 0 else '"fixed"')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass
    instance = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{instance.server_port}/sample.zip"
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join()


class RemoteZipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
            stream.writestr("fixture.txt", "small official-like archive fixture")
        cls.payload = archive.getvalue()

    def test_zip_metadata_and_crc_extraction_over_range(self):
        with server(self.payload) as url:
            reader = RangeZip(url, max_bytes=4096)
            with zipfile.ZipFile(reader) as archive:
                self.assertEqual(archive.read("fixture.txt"), b"small official-like archive fixture")
            self.assertLess(reader.transferred, 4096)
            self.assertEqual(reader.size, len(self.payload))

    def test_server_ignoring_range_is_rejected(self):
        with server(self.payload, ranges=False) as url:
            with self.assertRaisesRegex(ValueError, "full download refused"):
                RangeZip(url)

    def test_cap_prevents_large_request(self):
        with server(self.payload) as url:
            reader = RangeZip(url, max_bytes=4)
            with self.assertRaisesRegex(ValueError, "budget exceeded"):
                reader.read(4)
            self.assertEqual(reader.transferred, 1)

    def test_etag_change_rejected(self):
        with server(self.payload, wrong_etag=True) as url:
            reader = RangeZip(url)
            reader.seek(10)
            with self.assertRaisesRegex(ValueError, "archive changed"):
                reader.read(4)

    def test_bad_content_range_rejected(self):
        with server(self.payload, wrong_range=True) as url:
            with self.assertRaises(ValueError):
                RangeZip(url)
