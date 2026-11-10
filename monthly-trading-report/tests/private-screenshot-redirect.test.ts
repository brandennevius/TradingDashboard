import assert from "node:assert/strict";
import test from "node:test";
import {mockR2} from "./fixtures/mock-r2";
import {privateScreenshotRedirect,shouldRedirectScreenshot} from "../lib/screenshot-storage";

test("image views redirect while fetches and full-backup downloads stay same-origin",()=>{
  const prior=process.env.SCREENSHOT_DIRECT_SERVING;
  process.env.SCREENSHOT_DIRECT_SERVING="enabled";
  try {
    const request=(headers:Record<string,string>,path="/")=>new Request(`https://app.test${path}`,{headers});
    assert.equal(shouldRedirectScreenshot(request({'sec-fetch-dest':'image'})),true);
    assert.equal(shouldRedirectScreenshot(request({'sec-fetch-dest':'document'})),true);
    assert.equal(shouldRedirectScreenshot(request({'accept':'image/png'})),true);
    assert.equal(shouldRedirectScreenshot(request({'sec-fetch-dest':'empty','accept':'*/*'})),false);
    assert.equal(shouldRedirectScreenshot(request({'accept':'*/*'})),false);
    assert.equal(shouldRedirectScreenshot(request({'sec-fetch-dest':'image'},'/?proxy=1')),false);
  } finally {if(prior===undefined)delete process.env.SCREENSHOT_DIRECT_SERVING;else process.env.SCREENSHOT_DIRECT_SERVING=prior;}
});

test("private redirects sign a one-minute object URL without downloading bytes",async()=>{
  const r2=mockR2();const previous=process.env.SCREENSHOT_DIRECT_SERVING;
  const hash="a".repeat(64);
  const source={storage_path:`trade_screenshots/shot/${hash}`,image_sha256:hash,image_bytes:123,mime_type:"image/png",file_name:'chart"\r\n.png'};
  try {
    delete process.env.SCREENSHOT_DIRECT_SERVING;
    assert.equal(await privateScreenshotRedirect(source),null);
    process.env.SCREENSHOT_DIRECT_SERVING="enabled";
    r2.controls.failReads=true;
    const response=await privateScreenshotRedirect(source);
    assert.equal(response!.status,307);assert.equal(await response!.text(),"");
    assert.equal(response!.headers.get('cache-control'),'private, no-store');
    const url=new URL(response!.headers.get('location')!);
    assert.equal(url.protocol,"https:");assert.equal(url.searchParams.get('X-Amz-Expires'),'60');
    assert.equal(url.searchParams.get('response-content-type'),'image/png');
    assert.equal(url.searchParams.get('response-cache-control'),'private, no-store');
    assert.equal(url.searchParams.get('response-content-disposition'),'inline; filename="chart___.png"');
    assert.equal(r2.objects.size,0);
    assert.equal(await privateScreenshotRedirect({...source,storage_path:null}),null);
    assert.equal(await privateScreenshotRedirect({...source,mime_type:'image/svg+xml'}),null);
    await assert.rejects(privateScreenshotRedirect({...source,storage_path:'../bad'}),/Invalid/);
    await assert.rejects(privateScreenshotRedirect({...source,image_sha256:'b'.repeat(64)}),/integrity/);
  } finally {r2.restore();if(previous===undefined)delete process.env.SCREENSHOT_DIRECT_SERVING;else process.env.SCREENSHOT_DIRECT_SERVING=previous;}
});
