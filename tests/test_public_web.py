import http.client
import importlib.util
import io
import json
import tempfile
import threading
import unittest
import socket
from pathlib import Path
from unittest.mock import Mock

from journal_suggester.public_app import PublicApplication, RateLimit, waitress_proxy_options
from journal_suggester.paper_import import (ARXIV_SCOPE_MESSAGE, ArxivScopeError,
                                          ARXIV_UNAVAILABLE_MESSAGE, ArxivUnavailableError)
from journal_suggester.web_rpc import import_metadata, make_rpc_server, read_message, rpc


class PublicWebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'host').write_text('journal.example.com')
        self.backend = Mock(return_value={'suggestions': []})
        self.app = PublicApplication(self.root / 'host', '/model', '/metadata', self.root, self.backend)
        self.query = {'title': 'Graphs and spectral gaps', 'abstract':
                      'We establish new estimates for mixing times on sparse random regular graphs.'}

    def request(self, path='/', body=None, **changes):
        raw = json.dumps(body).encode() if body is not None else b''
        env = {'REQUEST_METHOD': 'POST' if body is not None else 'GET', 'PATH_INFO': path,
               'HTTP_HOST': 'journal.example.com', 'HTTP_ORIGIN': 'https://journal.example.com',
               'HTTP_CF_CONNECTING_IP': '203.0.113.1', 'CONTENT_TYPE': 'application/json',
               'CONTENT_LENGTH': str(len(raw)), 'wsgi.input': io.BytesIO(raw)}
        env.update(changes)
        response = []
        data = b''.join(self.app(env, lambda status, headers: response.extend([int(status.split()[0]), dict(headers)])))
        return *response, data

    def test_search_sends_only_validated_plaintext_to_model(self):
        query = self.query | {'command': 'cat /etc/passwd', 'title': 'A < B and C > D'}
        self.assertEqual(self.request('/api/suggest', query)[0], 200)
        payload = self.backend.call_args.args[1]
        self.assertEqual(payload['title'], 'A < B and C > D')
        self.assertNotIn('command', payload)

    def test_arxiv_only_passes_identifier(self):
        self.assertEqual(self.request('/api/import/arxiv', {'url': 'https://arxiv.org/abs/2301.00001v2'})[0], 200)
        self.backend.assert_called_once_with('/metadata', {'url': '2301.00001v2'}, timeout=25)
        self.backend.reset_mock()
        for url in ['http://192.168.1.1', 'file:///etc/passwd', 'https://arxiv.org/abs/2301.00001?x=1']:
            self.assertEqual(self.request('/api/import/arxiv', {'url': url})[0], 400)
        self.backend.assert_not_called()

    def test_arxiv_scope_error_is_specific_and_model_errors_stay_private(self):
        self.backend.return_value = {'error': 'arxiv_out_of_scope'}
        status, _, raw = self.request('/api/import/arxiv', {'url': '2301.00001', 'categories': ['math.PR']})
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(raw), {'error': ARXIV_SCOPE_MESSAGE})
        self.backend.assert_called_once_with('/metadata', {'url': '2301.00001'}, timeout=25)
        self.assertEqual(self.request('/api/suggest', self.query)[0], 503)
        self.backend.return_value = {'suggestions': []}
        self.assertEqual(self.request('/api/suggest', self.query)[0], 200)

    def test_origin_host_proxy_and_content_guards(self):
        for change, status in [({'HTTP_HOST': 'evil.example.com'}, 403),
                               ({'HTTP_ORIGIN': 'https://evil.example.com'}, 403),
                               ({'HTTP_ORIGIN': ''}, 403),
                               ({'HTTP_CF_CONNECTING_IP': ''}, 403),
                               ({'HTTP_CF_CONNECTING_IP': '1.1.1.1,2.2.2.2'}, 403),
                               ({'HTTP_SEC_FETCH_SITE': 'cross-site'}, 403),
                               ({'CONTENT_TYPE': 'multipart/form-data'}, 415),
                               ({'CONTENT_LENGTH': '150001'}, 413),
                               ({'CONTENT_LENGTH': 'garbage'}, 413),
                               ({'HTTP_TRANSFER_ENCODING': 'chunked'}, 413)]:
            with self.subTest(change=change):
                self.assertEqual(self.request('/api/suggest', self.query, **change)[0], status)
        self.backend.assert_not_called()

    def test_arxiv_outage_has_a_specific_safe_message(self):
        self.backend.return_value = {'error': 'arxiv_unavailable'}
        status, _, raw = self.request('/api/import/arxiv', {'url': '2012.05485'})
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(raw), {'error': ARXIV_UNAVAILABLE_MESSAGE})

    def test_no_filesystem_or_private_api_routes(self):
        for path in ['/api/info', '/.git/config', '/etc/passwd', '/api/import/pdf',
                     '/methodology/records/../../configs/models.json', '/methodology/records/models.json']:
            self.assertEqual(self.request(path)[0], 404)
        status, headers, _ = self.request('/')
        self.assertEqual(status, 200)
        self.assertIn("frame-ancestors 'none'", headers['Content-Security-Policy'])
        self.assertEqual(self.request('/', REQUEST_METHOD='HEAD')[2], b'')

    def test_busy_model_does_not_queue_or_block_arxiv(self):
        self.app.model_lock.acquire()
        self.addCleanup(self.app.model_lock.release)
        self.assertEqual(self.request('/api/suggest', self.query)[0], 503)
        self.backend.assert_not_called()
        self.assertEqual(self.request('/api/import/arxiv', {'url': '2301.00001'})[0], 200)

    def test_error_details_and_submitted_text_never_returned(self):
        self.backend.side_effect = RuntimeError('/home/private SECRET manuscript')
        status, _, body = self.request('/api/suggest', self.query)
        self.assertEqual(status, 503)
        self.assertNotIn(b'SECRET', body)

    def test_ip_limits_ipv6_subnet_and_spoofed_forwarding(self):
        for _ in range(6):
            self.assertEqual(self.request('/api/suggest', self.query)[0], 200)
        status, headers, _ = self.request('/api/suggest', self.query, HTTP_X_FORWARDED_FOR='8.8.8.8')
        self.assertEqual(status, 429)
        self.assertEqual(headers['Retry-After'], '60')
        for i in range(6):
            self.assertEqual(self.request('/api/suggest', self.query, HTTP_CF_CONNECTING_IP=f'2001:db8::{i+1}')[0], 200)
        self.assertEqual(self.request('/api/suggest', self.query, HTTP_CF_CONNECTING_IP='2001:db8::ff')[0], 429)

    def test_rate_windows_global_budget_and_memory_bound(self):
        now = [0]
        limit = RateLimit(lambda: now[0])
        for i in range(12):
            self.assertTrue(limit.allow(str(i)))
        self.assertFalse(limit.allow('new'))
        now[0] = 60
        self.assertTrue(limit.allow('new'))
        for minute in range(1, 25):
            now[0] = minute * 60
            for i in range(12 - (minute == 1)):
                self.assertTrue(limit.allow(f'{minute}-{i}'))
        now[0] = 25 * 60
        self.assertFalse(limit.allow('global-hour-limit'))
        self.assertLessEqual(len(limit.clients), 300)
        now[0] = 7200
        self.assertTrue(limit.allow('reset'))
        self.assertEqual(len(limit.clients), 1)

    def test_missing_hostname_fails_closed(self):
        (self.root / 'host').unlink()
        self.assertEqual(self.request('/')[0], 503)

    def test_tailscale_uses_only_its_overwritten_client_header(self):
        self.app.proxy = 'tailscale'
        self.assertEqual(self.request('/api/suggest', self.query)[0], 403)
        for i in range(6):
            self.assertEqual(self.request('/api/suggest', self.query,
                                         HTTP_X_FORWARDED_FOR='203.0.113.3',
                                         HTTP_CF_CONNECTING_IP=f'198.51.100.{i+1}')[0], 200)
        self.assertEqual(self.request('/api/suggest', self.query,
                                     HTTP_X_FORWARDED_FOR='203.0.113.3',
                                     HTTP_CF_CONNECTING_IP='198.51.100.200')[0], 429)
        self.assertEqual(self.request('/api/suggest', self.query,
                                     HTTP_X_FORWARDED_FOR='1.1.1.1,2.2.2.2')[0], 403)


class RpcTests(unittest.TestCase):
    def test_metadata_scope_code_survives_rpc_without_exception_details(self):
        importer = Mock()
        importer.fetch.side_effect = ArxivScopeError()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'metadata.sock'
            with make_rpc_server(path, lambda body: import_metadata(importer, body)) as server:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    self.assertEqual(rpc(path, {'url': '2301.00001'}), {'error': 'arxiv_out_of_scope'})
                    importer.fetch.side_effect = ArxivUnavailableError()
                    self.assertEqual(rpc(path, {'url': '2012.05485'}), {'error': 'arxiv_unavailable'})
                    importer.fetch.side_effect = ValueError('/home/private SECRET')
                    self.assertEqual(rpc(path, {'url': '2301.00001'}), {'error': 'Request failed'})
                finally:
                    server.shutdown()
                    thread.join()

    def test_bounded_json_protocol_and_real_socket_roundtrip(self):
        for raw, limit in [(b'{}', 20), (b'[]\n', 20), (b'{}\n', 2), (b'bad\n', 20)]:
            with self.assertRaises(ValueError):
                read_message(io.BytesIO(raw), limit)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'api.sock'
            with make_rpc_server(path, lambda body: {'echo': body}) as server:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    self.assertEqual(rpc(path, {'title': '<math>'}), {'echo': {'title': '<math>'}})
                    self.assertEqual(path.stat().st_mode & 0o777, 0o660)
                finally:
                    server.shutdown()
                    thread.join()


@unittest.skipUnless(importlib.util.find_spec('waitress'), 'Install the web extra for production transport checks')
class ProductionTransportTests(unittest.TestCase):
    setUp = PublicWebTests.setUp

    def test_waitress_unix_socket_enforces_wire_limits_and_headers(self):
        from waitress import create_server
        path = self.root / 'http.sock'
        server = create_server(self.app, unix_socket=str(path), unix_socket_perms='660',
                               max_request_body_size=150000, max_request_header_size=16384,
                               connection_limit=32, backlog=32, threads=4)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            def send(body, **changes):
                conn = http.client.HTTPConnection('journal.example.com')
                conn.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                conn.sock.settimeout(5)
                conn.sock.connect(str(path))
                headers = {'Content-Type': 'application/json', 'Origin': 'https://journal.example.com',
                           'CF-Connecting-IP': '203.0.113.1'}
                headers.update(changes)
                conn.request('POST', '/api/suggest', body=body, headers=headers)
                response = conn.getresponse()
                result = response.status, response.read()
                conn.close()
                return result
            self.assertEqual(send(json.dumps(self.query))[0], 200)
            self.assertEqual(send(json.dumps(self.query), Origin='https://evil.example.com')[0], 403)
            self.assertEqual(send(b'x' * 150001)[0], 413)
            self.assertEqual(path.stat().st_mode & 0o777, 0o660)
        finally:
            server.close()
            thread.join(timeout=2)

    def test_real_funnel_header_shape_and_host_validation(self):
        from waitress import create_server
        self.app.proxy = 'tailscale'
        path = self.root / 'funnel.sock'
        server = create_server(self.app, unix_socket=str(path), unix_socket_perms='660',
                               **waitress_proxy_options('tailscale'))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            def request(**changes):
                conn = http.client.HTTPConnection('localhost')
                conn.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                conn.sock.settimeout(5)
                conn.sock.connect(str(path))
                headers = {'Content-Type': 'application/json', 'Origin': 'https://journal.example.com',
                           'X-Forwarded-Host': 'journal.example.com', 'X-Forwarded-Proto': 'https',
                           'X-Forwarded-For': '203.0.113.2', 'CF-Connecting-IP': 'attacker-controlled'}
                headers.update(changes)
                conn.request('POST', '/api/suggest', body=json.dumps(self.query), headers=headers)
                response = conn.getresponse()
                code = response.status
                response.read()
                conn.close()
                return code
            self.assertEqual(request(), 200)
            self.assertEqual(request(**{'X-Forwarded-Host': 'wrong.example.com'}), 403)
            self.assertEqual(request(**{'Origin': 'https://wrong.example.com'}), 403)
            self.assertEqual(request(**{'X-Forwarded-For': ''}), 403)
        finally:
            server.close()
            thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
