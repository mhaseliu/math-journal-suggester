import http.client
import io
import json
import tempfile
import threading
import unittest
import urllib.request
import urllib.response
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock, patch

from journal_suggester.app import make_server
from journal_suggester.io import write_jsonl
from journal_suggester.paper_import import (ArxivImporter, ArxivScopeError, ArxivUnavailableError,
                                          MAX_METADATA_BYTES, normalize_arxiv, parse_arxiv, parse_arxiv_html)
from journal_suggester.web_service import SearchService, citation, validate_query


MATH_CATEGORY = b'<category term="math.CO" scheme="http://arxiv.org/schemas/atom"/>'
ARXIV_FEED = b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2301.00001v2</id><title>A &lt; B</title><summary>We prove that x &lt; y for graphs.</summary>' + MATH_CATEGORY + b'</entry></feed>'
# Reduced official-page structure from the reported failing mathematics paper.
ARXIV_HTML = b'''<!DOCTYPE html><html><head>
<meta property="og:url" content="https://arxiv.org/abs/2012.05485v1" />
<meta name="citation_title" content="Triangles with Vertices Equidistant to a Pedal Triangle" />
<meta name="citation_arxiv_id" content="2012.05485" />
<meta name="citation_abstract" content="A &lt; B, $x^2$, and geometric constructions." />
</head><body><table><tr><td class="tablecell subjects">
<span class="primary-subject">Metric Geometry (math.MG)</span>
</td></tr></table></body></html>'''


class ImportTests(unittest.TestCase):
    def test_arxiv_formats_and_private_url_rejection(self):
        for url in ('2301.00001v2','https://arxiv.org/abs/2301.00001v2'):
            self.assertEqual(normalize_arxiv(url),'2301.00001v2')
        self.assertEqual(normalize_arxiv('https://arxiv.org/abs/2610.08776'),'2610.08776')
        self.assertEqual(normalize_arxiv('https://arxiv.org/abs/0706.0001'),'0706.0001')
        for url in ('http://127.0.0.1/a', 'https://arxiv.org.evil/abs/2301.00001',
                    'https://arxiv.org@localhost/abs/2301.00001','https://arxiv.org:9999/abs/2301.00001',
                    'file:///etc/passwd','https://arxiv.org/abs/../secret',None):
            with self.assertRaises(ValueError): normalize_arxiv(url)

    def test_arxiv_checks_identity_and_preserves_math(self):
        xml = ARXIV_FEED
        paper = parse_arxiv(xml,'2301.00001')
        self.assertEqual(paper['title'],'A < B')
        self.assertIn('x < y',paper['abstract'])
        for ident in ('2301.00002','2301.00001v1'):
            with self.assertRaises(ValueError): parse_arxiv(xml,ident)
        self.assertEqual(parse_arxiv(xml.replace(b'http://arxiv.org',b'https://arxiv.org'),'2301.00001')['arxiv_id'],'2301.00001v2')
        with self.assertRaises(ValueError):
            parse_arxiv(xml.replace(b'http://arxiv.org/abs/',b'https://evil.example/abs/'),'2301.00001')

    def test_math_primary_crosslisting_and_aliases_qualify(self):
        for category in ('math.PR', 'math.MP', 'math-ph', 'cs.IT', 'cs.NA', 'stat.TH'):
            xml = ARXIV_FEED.replace(b'math.CO', category.encode())
            with self.subTest(category=category):
                self.assertEqual(parse_arxiv(xml, '2301.00001')['arxiv_id'], '2301.00001v2')
        primary = b'<x:primary_category xmlns:x="http://arxiv.org/schemas/atom" term="hep-th"/>'
        xml = ARXIV_FEED.replace(MATH_CATEGORY, primary + MATH_CATEGORY)
        self.assertEqual(parse_arxiv(xml, '2301.00001')['title'], 'A < B')
        primary_math = primary.replace(b'hep-th', b'math.PR')
        self.assertEqual(parse_arxiv(ARXIV_FEED.replace(MATH_CATEGORY, primary_math), '2301.00001')['title'], 'A < B')

    def test_non_math_missing_and_misleading_categories_rejected(self):
        for category in ('hep-th', 'astro-ph.HE', 'cs.LG', 'physics.optics', 'stat.ML', 'math.', 'math.PR.extra'):
            xml = ARXIV_FEED.replace(b'math.CO', category.encode())
            with self.subTest(category=category), self.assertRaises(ArxivScopeError):
                parse_arxiv(xml, '2301.00001')
        for category in (b'', MATH_CATEGORY.replace(b'http://arxiv.org/schemas/atom', b'http://example.org/msc'),
                         MATH_CATEGORY.replace(b' scheme="http://arxiv.org/schemas/atom"', b'')):
            with self.assertRaisesRegex(ValueError, 'subject categories'):
                parse_arxiv(ARXIV_FEED.replace(MATH_CATEGORY, category), '2301.00001')
        xml = ARXIV_FEED.replace(b'math.CO', b'hep-th').replace(b'We prove', b'math.CO math.PR We prove')
        with self.assertRaises(ArxivScopeError):
            parse_arxiv(xml, '2301.00001')

    def test_rejected_import_is_not_cached_or_returned(self):
        importer = ArxivImporter()
        requests, transport = self.transport(body=ARXIV_FEED.replace(b'math.CO', b'hep-th'))
        with transport, self.assertRaises(ArxivScopeError):
            importer.fetch('2301.00001')
        self.assertEqual(len(requests), 1)
        self.assertFalse(importer.cache)

    def test_bad_arxiv_inputs_never_fetch(self):
        inputs = ['http://10.0.0.1/', 'http://127.1/', 'http://[::1]/',
                  'http://arxiv.org/abs/2301.00001', 'https://arxiv.org/pdf/2301.00001',
                  'https://arxiv.org/html/2301.00001', 'https://www.arxiv.org/abs/2301.00001',
                  'https://export.arxiv.org/abs/2301.00001', 'arxiv.org/abs/2301.00001',
                  'https://arxiv.org/abs/math.AG/0301234', 'arXiv:2301.00001',
                  '2300.00001', '2313.00001', '2301.000', '2301.000001', '2301.00001v0',
                  'https://arxiv.org/abs/2301.00001/', 'https://arxiv.org/abs/2301.00001.pdf',
                  'https://arxiv.org.evil/abs/2301.00001', 'https://evil@arxiv.org/abs/2301.00001',
                  'https://arxiv.org:443/abs/2301.00001', '//arxiv.org/abs/2301.00001',
                  'https://arxiv.org/abs/2301.00001?url=http://localhost',
                  'https://arxiv.org/abs/2301.00001#fragment', 'https://arxiv.org/abs/%32%33%30%31.00001',
                  'https://arxiv.org/abs/2301.00001/../../etc/passwd', 'https://arxiv.org\\@localhost/abs/2301.00001',
                  'https://arxiv.org/abs/2301.00001\n', 'https://arxiv.org/ab\ts/2301.00001',
                  '２３０１.00001', 'file:///etc/passwd', 'gopher://localhost/', '2301.00001;id',
                  '2301.00001,2301.00002', 'x'*301, None, {}]
        with patch('urllib.request.build_opener') as opener:
            for value in inputs:
                with self.subTest(value=value), self.assertRaises(ValueError):
                    ArxivImporter().fetch(value)
            opener.assert_not_called()

    def transport(self, body=ARXIV_FEED, status=200, fallback=None):
        # Exercise urllib's real handler chain without making any network requests.
        requests=[]
        class FixtureTransport(urllib.request.BaseHandler):
            handler_order=100
            def https_open(self, request):
                requests.append(request.full_url)
                response_status, response_body = status, body
                if fallback is not None and len(requests) > 1:
                    response_status, response_body = fallback
                if isinstance(response_body, Exception):
                    raise response_body
                headers=Message()
                headers['Content-Type']='application/atom+xml'
                if response_status != 200: headers['Location']='http://127.0.0.1/private'
                response=urllib.response.addinfourl(io.BytesIO(response_body),headers,request.full_url,response_status)
                response.msg='OK' if response_status==200 else 'Fixture error'
                return response
            http_open=https_open
        build=urllib.request.build_opener
        return requests, patch('urllib.request.build_opener',side_effect=lambda *handlers:
                               build(urllib.request.ProxyHandler({}),*handlers,FixtureTransport()))

    def test_api_failure_falls_back_only_to_same_official_abstract_page_and_caches(self):
        for failure, status in ((TimeoutError(), 200), (urllib.error.URLError('offline'), 200), (b'', 503)):
            requests, transport = self.transport(body=failure, status=status, fallback=(200, ARXIV_HTML))
            importer = ArxivImporter()
            with self.subTest(failure=failure), transport:
                result = importer.fetch('https://arxiv.org/abs/2012.05485v1')
                self.assertEqual(result['arxiv_id'], '2012.05485v1')
                self.assertEqual(importer.fetch('2012.05485v1'), result)
            self.assertEqual(requests, ['https://export.arxiv.org/api/query?id_list=2012.05485v1',
                                        'https://arxiv.org/abs/2012.05485v1'])

    def test_fallback_keeps_scope_redirect_and_size_checks(self):
        for body, status, exception in ((ARXIV_HTML.replace(b'math.MG', b'gr-qc'), 200, ArxivScopeError),
                                        (b'', 302, ValueError),
                                        (b'x' * (MAX_METADATA_BYTES + 1), 200, ValueError),
                                        (TimeoutError(), 200, ArxivUnavailableError)):
            requests, transport = self.transport(body=TimeoutError(), fallback=(status, body))
            importer = ArxivImporter()
            with self.subTest(status=status, exception=exception), transport, self.assertRaises(exception):
                importer.fetch('2012.05485')
            self.assertEqual(requests, ['https://export.arxiv.org/api/query?id_list=2012.05485',
                                        'https://arxiv.org/abs/2012.05485'])
            self.assertFalse(importer.cache)

    def test_access_denials_and_rate_limits_do_not_fall_back(self):
        for status in (401, 403, 404, 429):
            requests, transport = self.transport(status=status)
            with self.subTest(status=status), transport, self.assertRaises(ArxivUnavailableError):
                ArxivImporter().fetch('2012.05485')
            self.assertEqual(len(requests), 1)

    def test_html_preserves_plaintext_and_checks_paper_and_version(self):
        result = parse_arxiv_html(ARXIV_HTML, '2012.05485')
        self.assertEqual(result['abstract'], 'A < B, $x^2$, and geometric constructions.')
        self.assertEqual(result['source_url'], 'https://arxiv.org/abs/2012.05485v1')
        for ident in ('2012.05486', '2012.05485v2'):
            with self.subTest(ident=ident), self.assertRaises(ValueError):
                parse_arxiv_html(ARXIV_HTML, ident)
        for original, replacement in ((b'content="2012.05485"', b'content="2012.05486"'),
                                       (b'content="2012.05485"', b'content="2012.05485v2"'),
                                       (b'https://arxiv.org/abs/', b'https://evil.example/abs/'),
                                       (b'2012.05485v1', b'2012.05485')):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                parse_arxiv_html(ARXIV_HTML.replace(original, replacement), '2012.05485')

    def test_html_math_crosslisting_aliases_and_category_spoofing(self):
        for category in (b'math.PR', b'math-ph', b'cs.IT', b'cs.NA', b'stat.TH', b'hep-th); Combinatorics (math.CO'):
            with self.subTest(category=category):
                self.assertEqual(parse_arxiv_html(ARXIV_HTML.replace(b'math.MG', category), '2012.05485')['arxiv_id'],
                                 '2012.05485v1')
        for category in (b'gr-qc', b'cs.LG', b'math.', b'math.PR.extra'):
            html = ARXIV_HTML.replace(b'math.MG', category).replace(b'geometric constructions', b'(math.MG)')
            with self.subTest(category=category), self.assertRaises(ValueError):
                parse_arxiv_html(html, '2012.05485')
        invalid = [ARXIV_HTML.replace(b'tablecell subjects', b'tablecell abstract'),
                   ARXIV_HTML.replace(b'</td>', b''),
                   ARXIV_HTML + b'<td class="subjects">Geometry (math.MG)</td>',
                   ARXIV_HTML + b'<meta name="citation_arxiv_id" content="2012.05486">',
                   ARXIV_HTML.replace(b'A &lt; B', b'x' * 30001),
                   ARXIV_HTML.decode().encode('utf-16'), b'\xff']
        for raw in invalid:
            with self.subTest(raw=raw[:30]), self.assertRaises(ValueError):
                parse_arxiv_html(raw, '2012.05485')

    def test_import_fetches_only_fixed_metadata_endpoint_and_caches(self):
        requests, transport=self.transport()
        importer=ArxivImporter()
        with transport:
            result=importer.fetch('https://arxiv.org/abs/2301.00001v2')
            result['title']='edited'
            self.assertEqual(importer.fetch('2301.00001v2')['title'],'A < B')
        self.assertEqual(requests,['https://export.arxiv.org/api/query?id_list=2301.00001v2'])

    def test_redirects_never_reach_the_destination(self):
        for status in (301,302,303,307,308):
            requests, transport=self.transport(status=status)
            with self.subTest(status=status), transport, self.assertRaises(ValueError):
                ArxivImporter().fetch('2301.00001')
            self.assertEqual(requests,['https://export.arxiv.org/api/query?id_list=2301.00001'])

    def test_metadata_size_and_xml_declaration_limits(self):
        requests, transport=self.transport(body=b'x'*(MAX_METADATA_BYTES+1))
        with transport, self.assertRaisesRegex(ValueError,'large'):
            ArxivImporter().fetch('2301.00001')
        for raw in (b'<!DOCTYPE feed [<!ENTITY x SYSTEM "file:///etc/passwd">]>'+ARXIV_FEED,
                    ARXIV_FEED.decode().encode('utf-16'), b'\xff',
                    ARXIV_FEED.replace(b'A &lt; B',b'x'*2001),
                    ARXIV_FEED.replace(b'We prove that x &lt; y for graphs.',b'x'*30001)):
            with self.subTest(raw=raw[:30]), self.assertRaises(ValueError): parse_arxiv(raw,'2301.00001')

    def test_busy_import_is_rejected_without_starting_another_fetch(self):
        importer=ArxivImporter()
        importer.lock.acquire()
        try:
            with patch('urllib.request.build_opener') as opener, self.assertRaises(RuntimeError):
                importer.fetch('2301.00001')
            opener.assert_not_called()
        finally:
            importer.lock.release()

    def test_user_math_is_plain_text(self):
        query=validate_query({'title':'A<B and C>D','abstract':'We prove that x<y and z>w for sufficiently large finite graphs.'})
        self.assertEqual(query['title'],'A<B and C>D')
        self.assertIn('x<y',query['abstract'])
        for value in ([],{}, {'title':'ABC','abstract':'tiny'}):
            with self.assertRaises(ValueError): validate_query(value)


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.refs=[{'paper_id':str(i),'title':f'Sparse random graphs {i}', 'abstract':'We study spectral gaps in sparse random graphs and mixing times.',
                   'journal_id':'journal-of-algebra' if i<3 else 'journal-of-number-theory','year':2020,
                   'doi':f'10.1234/{i}', 'arxiv_id':f'2301.0000{i}'} for i in range(6)]
        write_jsonl(Path(cls.temp.name)/'reference.jsonl',cls.refs)
        from journal_suggester.io import journals
        ranker=MagicMock()
        ranker.recommend.return_value={j['journal_id']:1/95 for j in journals()}
        cls.service=SearchService(ranker=ranker)
        cls.server=make_server(cls.service,port=0)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True); cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join(); cls.temp.cleanup()

    def request(self,path,body=None,headers=None):
        conn=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
        conn.request('POST' if body is not None else 'GET',path,body,headers or {})
        response=conn.getresponse(); result=(response.status,dict(response.getheaders()),response.read()); conn.close(); return result



    def test_unsafe_citation_link_not_renderable(self):
        self.assertEqual(citation({'url':'javascript:alert(1)'})['url'],'')

    def test_static_api_and_browser_security(self):
        status,headers,data=self.request('/')
        self.assertEqual(status,200); self.assertIn(b'id="manuscript-form"',data)
        self.assertNotIn(b'type="file"',data)
        self.assertIn("frame-ancestors 'none'",headers['Content-Security-Policy'])
        code,headers,page=self.request('/methodology')
        self.assertEqual(code,200)
        self.assertIn('text/html',headers['Content-Type'])
        self.assertIn(b'id="methodology-heading"',page)
        code,headers,record=self.request('/methodology/records/tuning.json')
        self.assertEqual(code,200)
        self.assertFalse(json.loads(record)['test_scored'])
        self.assertEqual(self.request('/methodology/records/../../../artifacts/training-8000-job.json')[0],404)
        self.assertEqual(self.request('/methodology/records/training-8000-job.json')[0],404)
        self.assertEqual(self.request('/api/info')[0],200)
        self.assertEqual(self.request('/../../.env')[0],404)
        self.assertEqual(self.request('/',headers={'Host':'evil.example'})[0],403)
        self.assertEqual(self.request('/api/info',headers={'Origin':'https://evil.example'})[0],403)

    def test_http_suggest_and_request_validation(self):
        query={'title':'A new study','abstract':'We study sparse random graphs and prove spectral gap bounds.'}
        code,_,raw=self.request('/api/suggest',json.dumps(query),{'Content-Type':'application/json'})
        self.assertEqual(code,200); self.assertEqual(len(json.loads(raw)['suggestions']),5)
        self.assertEqual(self.request('/api/suggest','{}',{'Content-Type':'text/plain'})[0],415)
        self.assertEqual(self.request('/api/suggest','[]',{'Content-Type':'application/json'})[0],400)
        self.assertEqual(self.request('/api/import/pdf',b'%PDF-broken',{'Content-Type':'application/pdf'})[0],404)
        self.assertEqual(self.request('/api/import/arxiv',b'%PDF-broken',{'Content-Type':'application/pdf'})[0],415)
        self.assertEqual(self.request('/api/suggest','{}',{'Content-Length':'99999999','Content-Type':'application/json'})[0],413)





if __name__ == '__main__': unittest.main()
