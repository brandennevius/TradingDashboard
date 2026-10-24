import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {createRequire} from 'node:module';
import test from 'node:test';
const require = createRequire(import.meta.url);
const parser = require('../public/cam-journal/broker-csv-parser.js');
const source = fs.readFileSync(new URL('../public/cam-journal/app.js', import.meta.url), 'utf8');
const html = fs.readFileSync(new URL('../public/cam-journal/index.html', import.meta.url), 'utf8');
function functionSource(name) {
  const start = source.indexOf(`function ${name}(`);
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
  const inputs = new Map();
  const context = vm.createContext({
    setups: [], trades: [], crypto: {randomUUID: () => 'test-id'},
    document: {querySelectorAll: () => []},
    $: id => { if (!inputs.has(id)) inputs.set(id, {value: ''}); return inputs.get(id); }
  });
  ['average', 'hasDefinedR', 'getRMultiple', 'validRValue', 'getWinLoss', 'deepClone', 'getSetupByName',
    'analyticsNumber', 'analyticsSummary', 'analyticsFieldsForTrade', 'readTradeAnalytics', 'analyticsTradeVersion',
    'analyticsFilteredTrades', 'setupSnapshot'].forEach(name => vm.runInContext(functionSource(name), context));
  return context;
}
const closed = (pl, risk, analytics) => ({pl, risk, analytics, status: 'Closed'});

test('analytics excludes missing excursions/R and open trades, while retaining real zero values', () => {
  const c = harness();
  const stats = c.analyticsSummary([
    closed(200, 100, {mfe: 2, mae: -1, directionConfirmed: 'yes', setupOutcome: 'Worked - Captured'}),
    closed(-50, '', {mfe: '', mae: null, directionConfirmed: 'no', setupOutcome: 'Failed - Managed'}),
    closed(0, 100, {mfe: 0, mae: 0}),
    {pl: 999, risk: 1, status: 'Open', analytics: {mfe: 999, mae: -999, directionConfirmed: 'yes'}}
  ]);
  assert.equal(stats.trades, 4);
  assert.equal(stats.closed, 3);
  assert.equal(stats.wins, 1);
  assert.equal(stats.rCount, 2);
  assert.equal(stats.avgR, 1);
  assert.equal(stats.avgMfe, 1);
  assert.equal(stats.avgMae, -0.5);
  assert.equal('directionReviewed' in stats, false);
  assert.equal('outcomeReviewed' in stats, false);
});

test('analytics with no measurements shows missing values rather than zero', () => {
  const c = harness();
  const s = c.analyticsSummary([closed(10, '', {mfe: '', mae: '  '})]);
  assert.equal(s.avgR, null);
  assert.equal(s.avgMfe, null);
  assert.equal(s.avgMae, null);
});

test('historical empty field snapshots do not inherit later questions', () => {
  const c = harness();
  c.setups = [{id: 's', name: 'BBC', currentVersion: 2, analyticsFields: [{id: 'new'}], versions: [{version: 1}, {version: 2, analyticsFields: [{id: 'new'}]}]}];
  assert.equal(c.analyticsFieldsForTrade({setup: 'BBC', analytics: {fieldsSnapshot: []}}).length, 0);
  assert.equal(c.analyticsFieldsForTrade({setup: 'BBC', setupScore: {setupVersion: 1}}).length, 0);
  assert.equal(c.analyticsFieldsForTrade({setup: 'BBC'})[0].id, 'new');
});

test('saved analytics fields and versions remain frozen after the setup is edited', () => {
  const c = harness();
  c.setups = [{id: 's', name: 'BBC', currentVersion: 2, versions: [{version: 1, analyticsFields: [{id: 'v1'}]}, {version: 2, analyticsFields: [{id: 'v2'}]}]}];
  const t = {setup: 'BBC', analytics: {setupVersion: 1, fieldsSnapshot: [{id: 'old', label: 'Original'}]}};
  c.$('editSetup').value = 'BBC';
  const saved = c.readTradeAnalytics(t);
  assert.equal(saved.setupVersion, 1);
  assert.equal(saved.fieldsSnapshot[0].id, 'old');
  assert.equal(saved.mfe, '');
  assert.equal(c.analyticsFieldsForTrade({setup: 'BBC', analytics: {setupVersion: 1}})[0].id, 'v1');
});

test('saving a review removes the retired hard-coded analytics fields', () => {
  const c = harness();
  c.setups = [{id: 's', name: 'BBC', currentVersion: 1, analyticsFields: []}];
  c.$('editSetup').value = 'BBC';
  const saved = c.readTradeAnalytics({
    setup: 'BBC',
    analytics: {
      directionConfirmed: 'yes',
      setupOutcome: 'Worked - Captured',
      entryQuality: 'Ideal',
      setupFollowed: 'yes'
    }
  });
  assert.equal(saved.directionConfirmed, undefined);
  assert.equal(saved.setupOutcome, undefined);
  assert.equal(saved.entryQuality, undefined);
  assert.equal(saved.setupFollowed, undefined);
});

test('switching setups saves the destination fields and version', () => {
  const c = harness();
  c.setups = [{id: 'dest', name: 'New Setup', currentVersion: 3, versions: [{version: 3, analyticsFields: [{id: 'dest-field'}]}]}];
  c.$('editSetup').value = 'New Setup';
  const saved = c.readTradeAnalytics({setup: 'BBC', setupScore: {setupVersion: 1}, analytics: {setupVersion: 1, fieldsSnapshot: [{id: 'old'}]}});
  assert.equal(saved.setupVersion, 3);
  assert.equal(saved.fieldsSnapshot[0].id, 'dest-field');
});

test('CSV reconciliation retains analytics alongside screenshots and grading', () => {
  const analytics = {mfe: 2, mae: -1, fieldsSnapshot: [{id: 'field'}], custom: {field: 'yes'}, setupVersion: 2};
  const saved = parser.preserveManualFields({id: 'manual', analytics, setupScore: {grade: 'A'}, screenshots: [{url: '/shot'}]}, {id: 'import', pl: 20});
  assert.deepEqual(saved.analytics, analytics);
  assert.equal(saved.setupScore.grade, 'A');
  assert.equal(saved.screenshots[0].url, '/shot');
  assert.equal(saved.pl, 20);
});

test('CSV reconciliation retains the manual risk-distance reference', () => {
  const saved = parser.preserveManualFields({id: 'manual', riskDistance: '1.375'}, {id: 'import', riskDistance: ''});
  assert.equal(saved.riskDistance, '1.375');
});

test('setup snapshots deep-copy analytics without altering grading criteria', () => {
  const c = harness();
  const setup = {criteria: {stf: [{text: 'Original', max: 2}]}, gradeRules: [], analyticsFields: [{id: 'f', label: 'Original'}]};
  const snapshot = c.setupSnapshot(setup, 2);
  setup.analyticsFields[0].label = 'Changed';
  assert.equal(snapshot.analyticsFields[0].label, 'Original');
  assert.equal(snapshot.criteria.stf[0].text, 'Original');
});

test('analytics filters distinguish frozen review versions and direction', () => {
  const c = harness();
  c.trades = [{setup: 'BBC', direction: 'Long', grade: 'A', analytics: {setupVersion: 1}}, {setup: 'BBC', direction: 'Short', grade: 'A', analytics: {setupVersion: 2}}];
  c.$('analyticsSetup').value = 'BBC'; c.$('analyticsVersion').value = '2'; c.$('analyticsDirection').value = 'Short';
  assert.equal(c.analyticsFilteredTrades().length, 1);
  c.$('analyticsGrade').value = 'B';
  assert.equal(c.analyticsFilteredTrades().length, 0);
});

test('Version 79 UI uses custom analytics questions and shared Data Health filters', () => {
  assert.doesNotMatch(html, /id="edit(?:DirectionConfirmed|SetupOutcome|EntryQuality|SetupFollowed)"/);
  assert.match(html, /id="editRiskDistance"/);
  assert.match(source, /class="analyticsDragHandle"/);
  assert.match(functionSource('renderDataHealth'), /getReportFilteredTrades\(\)/);
});
