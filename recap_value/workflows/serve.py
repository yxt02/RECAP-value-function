#!/usr/bin/env python3
"""Serve a local report and its linked videos; loopback only, with byte-range seeking."""
import argparse
from functools import partial
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
import re


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Accept-Ranges','bytes')
        super().end_headers()
    def do_GET(self):
        requested=self.headers.get('Range')
        path=Path(self.translate_path(self.path))
        if not requested or path.suffix.lower() not in ('.mp4','.webm') or not path.is_file():
            return super().do_GET()
        match=re.fullmatch(r'bytes=(\d*)-(\d*)',requested.strip())
        length=path.stat().st_size
        if not match or not any(match.groups()):
            self.send_error(416);return
        lo,hi=match.groups()
        if lo:
            start=int(lo);end=min(int(hi) if hi else length-1,length-1)
        else:
            start=max(0,length-int(hi));end=length-1
        if start>end or start>=length:
            self.send_response(416);self.send_header('Content-Range',f'bytes */{length}');self.end_headers();return
        self.send_response(206);self.send_header('Content-Type','video/webm' if path.suffix.lower()=='.webm' else 'video/mp4')
        self.send_header('Content-Range',f'bytes {start}-{end}/{length}')
        self.send_header('Content-Length',str(end-start+1));self.end_headers()
        try:
            with path.open('rb') as stream:
                stream.seek(start);remaining=end-start+1
                while remaining:
                    block=stream.read(min(256*1024,remaining))
                    if not block:break
                    self.wfile.write(block);remaining-=len(block)
        except (BrokenPipeError,ConnectionResetError):pass


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');p.add_argument('--port',type=int,default=0)
    args=p.parse_args(argv);root=(Path(__file__).resolve().parents[2]/args.directory).resolve()
    if not (root/'index.html').is_file():raise FileNotFoundError('Render the report first')
    server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(Handler,directory=str(root)))
    url=f'http://127.0.0.1:{server.server_port}/index.html'
    (root/'preview-url.txt').write_text(url+'\n');print(url,flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__=='__main__':main()
