import {test,expect} from "@playwright/test";

test("actual Next list routes authenticate before 304 and revalidate after an edit",async({request})=>{
  const routes=["/api/trades","/api/journal/branden/trade-log"];
  for(const route of routes) expect((await request.get(route,{headers:{'if-none-match':'*'}})).status()).toBe(401);
  expect((await request.post('/api/login',{data:{userId:'branden',password:process.env.E2E_PASSWORD}})).status()).toBe(200);
  const initialTags=new Map<string,string>();
  for(const route of routes) {
    const first=await request.get(route);expect(first.status()).toBe(200);
    expect(first.headers()['cache-control']).toContain('private, no-cache');
    expect(first.headers()['vary'].toLowerCase()).toContain('cookie');
    const tag=first.headers()['etag'];expect(tag).toBeTruthy();initialTags.set(route,tag);
    const repeat=await request.get(route,{headers:{'if-none-match':tag}});
    expect(repeat.status()).toBe(304);expect(await repeat.body()).toHaveLength(0);
  }
  const key=crypto.randomUUID();
  const input={symbol:'CACHE',entryDate:'2026-11-10',screenshots:[],status:'OPEN'};
  const created=await request.post('/api/trades',{data:input,headers:{'idempotency-key':key}});
  expect(created.status()).toBe(200);const id=(await created.json()).trade.id;
  const retried=await request.post('/api/trades',{data:input,headers:{'idempotency-key':key}});
  expect((await retried.json()).trade.id).toBe(id);
  for(const route of routes) {
    const changed=await request.get(route,{headers:{'if-none-match':initialTags.get(route)!}});
    expect(changed.status()).toBe(200);expect((await changed.json()).trades.some((trade:{id:string})=>trade.id===id)).toBe(true);
  }
  expect((await request.get('/api/public/branden-open-positions')).status()).toBe(200);
  expect((await request.delete(`/api/trades/${encodeURIComponent(id)}`)).status()).toBe(200);
});

test("browser no-cache fetches keep complete JSON, see edits immediately and recheck logout",async({page})=>{
  await page.request.post('/api/login',{data:{userId:'branden',password:process.env.E2E_PASSWORD}});
  await page.goto('/api/session');
  const read=()=>page.evaluate(async()=>{
    const response=await fetch('/api/trades',{cache:'no-cache'});
    return {status:response.status,body:await response.json()};
  });
  const first=await read();const second=await read();
  expect(first.status).toBe(200);expect(second).toEqual(first);
  const created=await page.request.post('/api/trades',{data:{symbol:'CACHEBROWSER',entryDate:'2026-11-10'}});
  const id=(await created.json()).trade.id;
  try {
    const changed=await read();
    expect(changed.status).toBe(200);
    expect(changed.body.trades.some((trade:{id:string})=>trade.id===id)).toBe(true);
  } finally {await page.request.delete(`/api/trades/${encodeURIComponent(id)}`);}
  await page.request.post('/api/logout');
  expect((await read()).status).toBe(401);
});
