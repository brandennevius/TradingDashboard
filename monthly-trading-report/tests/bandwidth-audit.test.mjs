import assert from 'node:assert/strict';
import test from 'node:test';
import {payloadAudit,bandwidthModel} from '../scripts/bandwidth-audit.mjs';
test('payload audit measures screenshot strings without rewriting unrelated data',()=>{
  const payload={trades:[{screenshots:['data:image/png;base64,YWJj','/existing'],notes:'preserve'}]};
  const original=JSON.stringify(payload), result=payloadAudit(payload,100);
  assert.equal(result.inlineImages,1);assert.equal(result.inlineStringBytes,26);
  assert.equal(JSON.stringify(payload),original);
  assert.throws(()=>payloadAudit({}),/trades/);
});
test('model keeps missing residual usage explicitly incomplete',()=>{
  const base={routes:[{name:'list',requestsPerDay:10,responseBytes:1000,notModifiedShare:0.5,notModifiedBytes:100}]};
  assert.equal(bandwidthModel(base).complete,false);assert.equal(bandwidthModel(base).headroomGB,null);
  const result=bandwidthModel({...base,incomingMonthlyGB:2,otherMonthlyGB:1,spikeMonthlyGB:2.5});
  assert.equal(result.routes[0].dailyBytes,5500);assert.equal(result.subtotalGB,5.500165);
  assert.throws(()=>bandwidthModel({routes:[{requestsPerDay:1,responseBytes:1,notModifiedShare:2}]}));
});
