import {test,expect} from "@playwright/test";
import {Client} from "pg";
import {createServer} from "node:http";
import type {AddressInfo} from "node:net";

test("signed screenshot routes preserve owner checks, image rendering and same-origin downloads",async({page,browser})=>{
  test.skip(!process.env.E2E_SCREENSHOT_ROUTES,"Use --screenshot-routes with isolated PostgreSQL.");
  const imageHeaders={'sec-fetch-dest':'image'};
  const tradeUrl='/api/trades/trade-1/screenshots/trade-shot';
  expect((await page.request.get(tradeUrl,{headers:imageHeaders,maxRedirects:0})).status()).toBe(401);
  await page.request.post('/api/login',{data:{userId:'branden',password:process.env.E2E_PASSWORD}});
  for(const route of [tradeUrl,'/api/cam-journal/screenshots/shared-shot','/api/watchlists/screenshots/watch-shot']) {
    const response=await page.request.get(route,{headers:imageHeaders,maxRedirects:0});
    expect(response.status()).toBe(307);
    expect(response.headers()['cache-control']).toContain('private, no-store');
    expect(new URL(response.headers()['location']).searchParams.get('X-Amz-Expires')).toBe('60');
  }
  expect((await page.request.get('/api/trades/wrong/screenshots/trade-shot',{headers:imageHeaders})).status()).toBe(404);
  expect((await page.request.get('/api/watchlists/screenshots/shared-shot',{headers:imageHeaders})).status()).toBe(404);
  const cam=await browser.newContext({baseURL:'http://127.0.0.1:3100'});
  try {
    await cam.request.post('/api/login',{data:{userId:'cam',password:process.env.E2E_PASSWORD}});
    expect((await cam.request.get(tradeUrl,{headers:imageHeaders,maxRedirects:0})).status()).toBe(404);
  } finally {await cam.close();}
  const delegate=await browser.newContext({baseURL:'http://127.0.0.1:3100'});
  try {
    await delegate.request.post('/api/login',{data:{userId:'tim',password:'tim'}});
    expect((await delegate.request.get(tradeUrl,{headers:imageHeaders,maxRedirects:0})).status()).toBe(307);
  } finally {await delegate.close();}
  const png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAACXBIWXMAAAPoAAAD6AG1e1JrAAAADElEQVQImWMITSsHAAJGATN1EKEHAAAAAElFTkSuQmCC','base64');
  // Use a local cross-origin image server in place of a real R2 bucket.
  const imageServer=createServer((_request,response)=>{response.writeHead(200,{'content-type':'image/png'});response.end(png);});
  await new Promise<void>(resolve=>imageServer.listen(0,'127.0.0.1',resolve));
  const port=(imageServer.address() as AddressInfo).port;
  try {
    await page.route(`**${tradeUrl}`,async route=>{
      const response=await route.fetch({maxRedirects:0});expect(response.status()).toBe(307);
      await route.fulfill({response,headers:{...response.headers(),location:`http://127.0.0.1:${port}/image.png`}});
    });
    await page.goto('/api/session');
    await page.setContent(`<img src="${tradeUrl}" alt="private chart">`);
    await expect.poll(()=>page.getByAltText('private chart').evaluate((img:HTMLImageElement)=>img.naturalWidth)).toBe(1);
    await page.unroute(`**${tradeUrl}`);
  } finally {await new Promise<void>(resolve=>imageServer.close(()=>resolve()));}
  // A legacy bytea row also remains fetchable for backups with direct serving on.
  const db=new Client({connectionString:process.env.DATABASE_URL,ssl:false});await db.connect();
  try {await db.query("update trade_screenshots set storage_path=null where id='trade-shot'");}
  finally {await db.end();}
  const fetched=await page.evaluate(async url=>{
    const response=await fetch(url);return {status:response.status,redirected:response.redirected,bytes:(await response.arrayBuffer()).byteLength};
  },tradeUrl);
  expect(fetched).toEqual({status:200,redirected:false,bytes:png.length});
});
