import fs from 'node:fs/promises';
import { gzipSync, brotliCompressSync } from 'node:zlib';
import { pathToFileURL } from 'node:url';

export function payloadAudit(payload, referenceBytes = 160) {
  if (!Number.isInteger(referenceBytes) || referenceBytes < 1) throw new Error('referenceBytes must be a positive integer.');
  if (!Array.isArray(payload.trades)) throw new Error('Expected a trades array.');
  let images = 0, inlineStringBytes = 0;
  const projected = {...payload, trades: payload.trades.map(trade => ({...trade,
    screenshots: trade.screenshots?.map(value => {
      if (typeof value !== 'string' || !/^data:image\/[^;]+;base64,/.test(value)) return value;
      images++; inlineStringBytes += Buffer.byteLength(value);
      return '/api/trades/' + String(images).padStart(Math.max(1, referenceBytes - 12), 'x');
    })
  }))};
  const sizes = value => { const bytes = Buffer.from(JSON.stringify(value)); return {
    jsonBytes: bytes.length, gzipBytes: gzipSync(bytes).length, brotliBytes: brotliCompressSync(bytes).length
  }; };
  return {trades: payload.trades.length, inlineImages: images, inlineStringBytes,
    assumedReferenceBytes: referenceBytes, current: sizes(payload), projected: sizes(projected)};
}

export function bandwidthModel(scenario) {
  const routes = scenario.routes.map(route => {
    for (const key of ['requestsPerDay','responseBytes']) if (!Number.isFinite(route[key]) || route[key] < 0) throw new Error(`Invalid ${key}`);
    const unchanged = route.notModifiedShare ?? 0;
    if (unchanged < 0 || unchanged > 1) throw new Error('Invalid notModifiedShare');
    const dailyBytes = route.requestsPerDay * ((1-unchanged)*route.responseBytes + unchanged*(route.notModifiedBytes ?? 500));
    return {name:route.name, dailyBytes, monthlyGB: dailyBytes*30/1e9};
  });
  const missing = ['incomingMonthlyGB','otherMonthlyGB','spikeMonthlyGB'].filter(key => !Number.isFinite(scenario[key]));
  for (const key of ['incomingMonthlyGB','otherMonthlyGB','spikeMonthlyGB']) if (scenario[key] < 0) throw new Error(`Invalid ${key}`);
  const subtotalGB = routes.reduce((sum,route)=>sum+route.monthlyGB,0) + ['incomingMonthlyGB','otherMonthlyGB','spikeMonthlyGB'].reduce((sum,key)=>sum+(scenario[key]??0),0);
  return {routes, complete:missing.length===0, missing, subtotalGB, headroomGB:missing.length ? null : 10-subtotalGB};
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [mode,file,referenceBytes] = process.argv.slice(2);
  if (!['--payload','--model'].includes(mode) || !file) throw new Error('Usage: --payload response.json [referenceBytes] | --model scenario.json. For SQL use psql -X -f scripts/audit-trade-payload.sql.');
  const value=JSON.parse(await fs.readFile(file,'utf8'));
  console.log(JSON.stringify(mode==='--payload' ? payloadAudit(value, referenceBytes===undefined ? 160 : Number(referenceBytes)) : bandwidthModel(value),null,2));
}
