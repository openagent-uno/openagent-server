import assert from 'node:assert/strict';
import test from 'node:test';

import { resolveChromiumExecutable, SearchEngine } from '../dist/search-engine.js';

const hit = {
  title: 'Relevant result',
  url: 'https://example.test/result',
  description: 'A useful answer from the first engine.',
  timestamp: '2026-09-01T00:00:00.000Z',
};

test('later empty or failed engines cannot erase earlier search results', async () => {
  const previousForce = process.env.FORCE_MULTI_ENGINE_SEARCH;
  const previousQuality = process.env.ENABLE_RELEVANCE_CHECKING;
  process.env.FORCE_MULTI_ENGINE_SEARCH = 'true';
  process.env.ENABLE_RELEVANCE_CHECKING = 'false';

  const engine = new SearchEngine();
  engine.tryBrowserBingSearch = async () => [hit];
  engine.tryBrowserBraveSearch = async () => {
    throw new Error('browser unavailable');
  };
  engine.tryDuckDuckGoSearch = async () => [];
  engine.handleBrowserError = async () => {};

  try {
    const result = await engine.search({ query: 'specific long-tail query', numResults: 5 });
    assert.equal(result.engine, 'Browser Bing');
    assert.deepEqual(result.results, [hit]);
  } finally {
    if (previousForce === undefined) delete process.env.FORCE_MULTI_ENGINE_SEARCH;
    else process.env.FORCE_MULTI_ENGINE_SEARCH = previousForce;
    if (previousQuality === undefined) delete process.env.ENABLE_RELEVANCE_CHECKING;
    else process.env.ENABLE_RELEVANCE_CHECKING = previousQuality;
  }
});

test('Chromium resolution prefers operator config and falls back to system Chrome', () => {
  const present = new Set([
    '/opt/openagent/chrome',
    '/usr/bin/google-chrome',
  ]);
  const exists = (candidate) => present.has(candidate);

  assert.equal(
    resolveChromiumExecutable(
      { OPENAGENT_CHROME_BINARY: '/opt/openagent/chrome' },
      'linux',
      exists,
    ),
    '/opt/openagent/chrome',
  );
  assert.equal(
    resolveChromiumExecutable({}, 'linux', exists),
    '/usr/bin/google-chrome',
  );
  assert.equal(
    resolveChromiumExecutable({}, 'linux', () => false),
    undefined,
  );
});
