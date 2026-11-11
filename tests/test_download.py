"""The INCLUDE downloader must survive flaky connections (real failure seen on Kanak's PC:
'IncompleteRead' after 7 MB). A local server that cuts every response after 64 KB proves that
download() resumes with HTTP Range, retries automatically and still verifies the MD5."""

import hashlib
import http.server
import socket
import threading

import pytest

import training.download_include as dl

PAYLOAD = bytes(range(256)) * 2000  # 512 KB fake zip
CUT = 64 * 1024  # server drops the connection after this many bytes per request


class FlakyHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):  # noqa: N802
        start = 0
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
        body = PAYLOAD[start:]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body[:CUT])  # then hang up early → client sees IncompleteRead
        self.wfile.flush()
        self.connection.shutdown(socket.SHUT_RDWR)  # hard hang-up, like a dropped home connection
        self.close_connection = True


@pytest.fixture()
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FlakyHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/f.zip"
    srv.shutdown()


def test_resumes_through_broken_connections(server, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "ZIPS", tmp_path)
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)  # no real waiting in tests
    item = {"name": "f.zip", "url": server, "size": len(PAYLOAD), "md5": hashlib.md5(PAYLOAD).hexdigest()}
    path = dl.download(item, retries=3)
    assert path.read_bytes() == PAYLOAD
    assert (tmp_path / "f.zip.ok").exists()
    # a second call is instant (already verified)
    assert dl.download(item, retries=0) == path


def test_bad_md5_is_deleted(server, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "ZIPS", tmp_path)
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    item = {"name": "g.zip", "url": server, "size": len(PAYLOAD), "md5": "0" * 32}
    with pytest.raises(RuntimeError, match="MD5"):
        dl.download(item, retries=3)
    assert not (tmp_path / "g.zip").exists()
