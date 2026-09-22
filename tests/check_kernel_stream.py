"""Integration check for interrupted large-file retrieval; requires Kaggle client."""
import hashlib
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer
from tempfile import TemporaryDirectory
from threading import Thread

import requests
from kaggle.api.kaggle_api_extended import KaggleApi


BODY = b'rsna-stream-test' * 200000


class InterruptedDownload(BaseHTTPRequestHandler):
    first = True

    def log_message(self, *args):
        pass

    def do_GET(self):
        start = int(self.headers.get('Range', 'bytes=0-').split('=')[1].split('-')[0])
        self.send_response(206 if start else 200)
        self.send_header('Content-Length', str(len(BODY) - start))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('ETag', '"frozen-test"')
        self.end_headers()
        if InterruptedDownload.first:
            InterruptedDownload.first = False
            self.wfile.write(BODY[:1200000])
            self.wfile.flush()
            self.close_connection = True
        else:
            self.wfile.write(BODY[start:])


if __name__ == '__main__':
    with TCPServer(('127.0.0.1', 0), InterruptedDownload) as server, TemporaryDirectory() as tmp:
        Thread(target=server.serve_forever, daemon=True).start()
        with requests.get(f'http://127.0.0.1:{server.server_address[1]}/pixels', stream=True) as response:
            path = Path(tmp) / 'pixels.npy'
            KaggleApi().download_file(response, str(path), None, resume=True, quiet=True)
        assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(BODY).digest()
        assert not list(Path(tmp).glob('*.kaggle-partial'))
        server.shutdown()
        print('Interrupted transfer resumed; complete bytes and SHA-256 match.')
