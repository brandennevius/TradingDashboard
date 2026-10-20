import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

const source = fs.readFileSync(new URL('../public/cam-journal/app.js', import.meta.url), 'utf8');
function functionSource(name) {
  const start = source.indexOf(`async function ${name}(`) >= 0
    ? source.indexOf(`async function ${name}(`) : source.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `Missing ${name}`);
  let depth = 0;
  for (let i = source.indexOf('{', start); i < source.length; i++) {
    if (source[i] === '{') depth++;
    if (source[i] === '}') depth--;
    if (depth === 0) return source.slice(start, i + 1);
  }
  throw new Error(`Unclosed ${name}`);
}
function harness() {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) {
      const classes = new Set();
      elements.set(id, {
        value: '', innerHTML: '', textContent: '', listeners: {},
        classList: { add: c => classes.add(c), remove: c => classes.delete(c), contains: c => classes.has(c) },
        addEventListener(event, callback) { this.listeners[event] = callback; }
      });
    }
    return elements.get(id);
  }
  const context = vm.createContext({
    console: {error() {}}, $: element, feedbackTickets: [], activeFeedbackTicketId: '',
    feedbackScreenshots: [{name: 'draft.png'}],
    renderFeedbackThreadList() {}, renderFeedbackMessageList() {}, renderFeedbackReplyScreenshotGallery() {},
    feedbackStatusLabel: s => s, feedbackStatusClass: s => s,
    fetch: async () => ({ok: true, json: async () => ({tickets: [{id: 'old', title: 'Old issue'}]})})
  });
  ['activeFeedbackTicket', 'renderFeedbackThreadView', 'loadFeedbackTickets'].forEach(name => vm.runInContext(functionSource(name), context));
  element('feedbackTitle').value = 'Unsaved title';
  element('feedbackDetails').value = 'Unsaved report';
  return {context, element};
}

test('refresh keeps the new-ticket form and its draft visible when old tickets exist', async () => {
  const {context, element} = harness();
  await context.loadFeedbackTickets();
  await context.loadFeedbackTickets();
  assert.equal(context.activeFeedbackTicketId, '');
  assert.equal(element('feedbackNewTicketPanel').classList.contains('hidden'), false);
  assert.equal(element('feedbackTitle').value, 'Unsaved title');
  assert.equal(element('feedbackDetails').value, 'Unsaved report');
  assert.equal(context.feedbackScreenshots.length, 1);
});

test('refresh preserves an explicitly selected existing conversation', async () => {
  const {context, element} = harness();
  context.activeFeedbackTicketId = 'old';
  await context.loadFeedbackTickets();
  assert.equal(context.activeFeedbackTicketId, 'old');
  assert.equal(element('feedbackThreadPanel').classList.contains('hidden'), false);
});

test('a delayed response cannot switch away after the user starts a new draft', async () => {
  const {context, element} = harness();
  context.activeFeedbackTicketId = 'old';
  let resolve;
  context.fetch = () => new Promise(r => { resolve = r; });
  const pending = context.loadFeedbackTickets('old');
  context.activeFeedbackTicketId = '';
  resolve({ok: true, json: async () => ({tickets: [{id: 'old'}]})});
  await pending;
  assert.equal(context.activeFeedbackTicketId, '');
  assert.equal(element('feedbackNewTicketPanel').classList.contains('hidden'), false);
});

test('an explicit selection can open the newly submitted ticket', async () => {
  const {context} = harness();
  await context.loadFeedbackTickets('old');
  assert.equal(context.activeFeedbackTicketId, 'old');
});

test('a missing conversation returns to the preserved draft without selecting another ticket', async () => {
  const {context, element} = harness();
  context.activeFeedbackTicketId = 'removed';
  await context.loadFeedbackTickets();
  assert.equal(context.activeFeedbackTicketId, '');
  assert.equal(element('feedbackDetails').value, 'Unsaved report');
});

test('network errors do not clear or hide a draft', async () => {
  const {context, element} = harness();
  context.fetch = async () => { throw new Error('offline'); };
  await context.loadFeedbackTickets();
  assert.equal(context.activeFeedbackTicketId, '');
  assert.equal(element('feedbackDetails').value, 'Unsaved report');
  assert.equal(context.feedbackScreenshots.length, 1);
});

test('both New buttons preserve existing draft fields and attachments', () => {
  const {context, element} = harness();
  const start = source.indexOf("if ($('newFeedbackThreadBtn'))");
  const end = source.indexOf("document.querySelectorAll('.feedbackKindBtn')", start);
  vm.runInContext(source.slice(start, end), context);
  context.activeFeedbackTicketId = 'old';
  element('newFeedbackThreadBtn').listeners.click();
  element('backToFeedbackNewBtn').listeners.click();
  assert.equal(context.activeFeedbackTicketId, '');
  assert.equal(element('feedbackTitle').value, 'Unsaved title');
  assert.equal(context.feedbackScreenshots.length, 1);
});
