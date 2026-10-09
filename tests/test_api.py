import json, threading, unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from api import create_server

class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=create_server(lambda body: {'value':body['value']},port=0)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.url=f'http://127.0.0.1:{cls.server.server_port}'
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()
    def request(self,raw,content_type='application/json'):
        return urlopen(Request(self.url+'/predict',data=raw,
                              headers={'Content-Type':content_type}),timeout=5)
    def test_health(self):
        with urlopen(self.url+'/health',timeout=5) as response:
            self.assertEqual(json.load(response),{'status':'ok'})
    def test_json_roundtrip(self):
        with self.request(b'{"value":12}') as response:
            self.assertEqual(json.load(response),{'value':12})
    def test_reject_invalid_bodies(self):
        for raw in [b'[]',b'{bad',b'{"value":NaN}',b'{}']:
            with self.assertRaises(HTTPError) as error: self.request(raw)
            self.assertEqual(error.exception.code,400)
            error.exception.close()
    def test_reject_large_body(self):
        with self.assertRaises(HTTPError) as error: self.request(b' '*65537)
        self.assertEqual(error.exception.code,413);error.exception.close()
    def test_content_type(self):
        with self.assertRaises(HTTPError) as error:self.request(b'{}','text/plain')
        self.assertEqual(error.exception.code,415);error.exception.close()

if __name__=='__main__':unittest.main()
