"""Small local JSON API for demos; use a production server for public deployment."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json

MAX_BODY=65536

def create_server(predict, port=8000, host='127.0.0.1'):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Avoid logging request data in this demo.

        def respond(self, status, data):
            raw=json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(raw)))
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path=='/health': self.respond(200,{'status':'ok'})
            else: self.respond(404,{'error':'Use POST /predict or GET /health'})

        def do_POST(self):
            if self.path!='/predict':
                self.respond(404,{'error':'Route not found'}); return
            if self.headers.get_content_type()!='application/json':
                self.respond(415,{'error':'Content-Type must be application/json'}); return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=MAX_BODY:
                    self.respond(413,{'error':'Request must contain 1–65536 bytes'}); return
                self.connection.settimeout(5)
                body=self.rfile.read(size)
                if len(body)!=size: raise ValueError('Incomplete request body')
                def reject_constant(value): raise ValueError('Non-finite JSON numbers are not accepted')
                payload=json.loads(body,parse_constant=reject_constant)
                if not isinstance(payload,dict): raise ValueError('JSON body must be an object')
                self.respond(200,predict(payload))
            except (ValueError,TypeError,KeyError,UnicodeError) as error:
                self.respond(400,{'error':str(error)})
            except TimeoutError:
                self.respond(408,{'error':'Request body timed out'})
            except Exception:
                self.respond(500,{'error':'Internal error; inspect the local model configuration'})

    return ThreadingHTTPServer((host,port),Handler)

def serve(predict,port=8000):
    with create_server(predict,port) as server:
        print(f'Local API: http://127.0.0.1:{server.server_port} (Ctrl+C to stop)',flush=True)
        try: server.serve_forever()
        except KeyboardInterrupt: pass
