#!/usr/bin/env python3
# Copyright 2024 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

"""Serves the TrustMark JS demo so that ONNX Runtime Web can use several CPU threads.

WebAssembly only runs multithreaded in a cross-origin isolated page, which needs the
Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers that
`python -m http.server` does not send. Run it from the js directory:

    python3 server.py [port]           # http://localhost:8000/index.html
    python3 server.py [port] --https   # https://, with the certificate and key in server.pem
"""

import http.server
import ssl
import sys


class IsolatedHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()


def main():
    ports = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    port = int(ports[0]) if ports else 8000
    server = http.server.ThreadingHTTPServer(('localhost', port), IsolatedHandler)
    scheme = 'http'
    if '--https' in sys.argv[1:]:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain('server.pem')
        server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = 'https'
    print(f'Serving {scheme}://localhost:{port}/index.html', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
