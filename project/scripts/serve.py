"""Serve only the reviewable demo and artifacts, bound to loopback."""
import argparse
import http.server
import pathlib
import urllib.parse

ROOT = pathlib.Path(__file__).resolve().parents[1]

class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        relative = urllib.parse.unquote(urllib.parse.urlsplit(path).path).lstrip('/')
        if relative in ('', 'index.html'):
            return str(ROOT/'demo/index.html')
        target = (ROOT/relative).resolve()
        if target.is_relative_to(ROOT/'artifacts') and target.is_file():
            return str(target)
        return str(ROOT/'demo/__not_found__')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=7860)
    args = parser.parse_args()
    print(f'Football EgoX: http://127.0.0.1:{args.port}',flush=True)
    http.server.ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
