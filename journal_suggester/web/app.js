'use strict';
const $ = id => document.getElementById(id);
let busy = false, source = {};
function message(id, text, error = false) {
  const node = $(id); node.textContent = text; node.hidden = !text; node.classList.toggle('error', error);
  node.setAttribute('role', error ? 'alert' : 'status');
}
function setBusy(value) {
  busy = value;
  for (const id of ['suggest', 'import-arxiv', 'try-example', 'clear', 'paper-title', 'abstract', 'arxiv-url']) $(id).disabled = value;
}
function clearResults() {
  $('results').replaceChildren();
  document.querySelector('.results-section').hidden = true;
  message('search-status', '');
}
function edited() { source = {}; clearResults(); }
$('paper-title').addEventListener('input', edited); $('abstract').addEventListener('input', edited);
async function api(url, body) {
  const controller = new AbortController(); const timer = setTimeout(() => controller.abort(), url === '/api/suggest' ? 190000 : 30000);
  try {
    const response = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body), signal: controller.signal});
    const result = await response.json(); if (!response.ok) throw new Error(result.error || 'Request failed. Please try again.'); return result;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('The request took too long. Please try again or paste your abstract.');
    if (error instanceof TypeError) throw new Error('Cannot reach the website. Please try again shortly.');
    throw error;
  } finally { clearTimeout(timer); }
}
async function importPaper(url, body) {
  if (busy) return;
  setBusy(true); message('import-status', 'Fetching the arXiv title and abstract…');
  try {
    const paper = await api(url, body);
    $('paper-title').value = paper.title; $('abstract').value = paper.abstract;
    source = {arxiv_id: paper.arxiv_id || '', doi: paper.doi || ''};
    clearResults(); message('import-status', 'Title and abstract filled. Review them, then choose “Find journals”.');
  } catch (error) { message('import-status', error.message, true); }
  finally { setBusy(false); }
}
$('arxiv-form').addEventListener('submit', event => { event.preventDefault(); importPaper('/api/import/arxiv', {url: $('arxiv-url').value.trim()}); });
$('try-example').addEventListener('click', () => {
  if (busy) return;
  $('arxiv-url').value = $('try-example').dataset.arxiv;
  $('arxiv-form').requestSubmit();
});
$('clear').addEventListener('click', () => {
  $('manuscript-form').reset(); $('arxiv-form').reset(); source = {}; clearResults(); message('import-status', ''); $('paper-title').focus();
});
function node(tag, className, text) {const item = document.createElement(tag); if (className) item.className = className; if (text) item.textContent = text; return item;}
function render(suggestions) {
  $('results').replaceChildren();
  suggestions.forEach((journal, index) => {
    const card = node('article', 'journal-card');
    card.append(node('h3', '', `${index + 1}. ${journal.journal_name}`));
    $('results').append(card);
  });
}
$('manuscript-form').addEventListener('submit', async event => {
  event.preventDefault(); if (busy) return; setBusy(true);
  document.querySelector('.results-section').hidden = false;
  $('results').replaceChildren();
  document.querySelector('.results-section').setAttribute('aria-busy','true');
  message('search-status', 'Finding journals…');
  try {
    const query = {...source, title: $('paper-title').value.trim(), abstract: $('abstract').value.trim()};
    const result = await api('/api/suggest', query); render(result.suggestions);
    message('search-status', result.suggestions.length ? '' : result.notice || 'No suggestions found. Try a more detailed abstract.');
    $('results-heading').focus({preventScroll: true});
    $('results-heading').scrollIntoView({behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block:'start'});
  } catch (error) { $('results').replaceChildren(); message('search-status', error.message, true); }
  finally {setBusy(false); document.querySelector('.results-section').setAttribute('aria-busy','false');}
});
