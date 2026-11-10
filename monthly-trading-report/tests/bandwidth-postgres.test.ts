import assert from "node:assert/strict";
import {randomUUID} from "node:crypto";
import test from "node:test";
import {Client} from "pg";
import sharp from "sharp";
import {mockR2} from "./fixtures/mock-r2";
import {trade} from "./fixtures/trade-review-export";
import {writeTradeWithInlineScreenshots} from "../lib/inline-trade-screenshots";

test("inline extraction commits exact originals atomically; new trade and upload retries are idempotent",{skip:!process.env.BACKUP_TEST_DATABASE_URL},async()=>{
  const target=new URL(process.env.BACKUP_TEST_DATABASE_URL!);
  assert.equal(target.hostname,"localhost");assert.match(target.pathname,/^\/trading_dashboard_backup_test/);
  const admin=new Client({connectionString:target.toString(),ssl:false});await admin.connect();
  const schema=`bandwidth_${randomUUID().replaceAll('-','')}`;
  const previous={...process.env};const r2=mockR2();let pool;
  try {
    await admin.query(`create schema ${schema}`);target.searchParams.set("options",`-c search_path=${schema}`);
    process.env.DATABASE_URL=target.toString();process.env.SCREENSHOT_STORAGE_WRITES="enabled";
    const store=await import("../lib/store");pool=store.getPool()!;
    const original=Buffer.from([0,255,128,1,2,3]);const inline=`data:image/png;base64,${original.toString('base64')}`;
    const input=trade({userId:"branden",screenshots:[inline],importSource:"",importRowKey:""});
    const key=randomUUID();
    const first=await store.upsertTrade(input,key);
    assert.ok(!first.trade.screenshots[0].startsWith('data:'));
    const images=await pool.query("select * from trade_screenshots");
    assert.equal(images.rowCount,1);assert.deepEqual(r2.objects.get(images.rows[0].storage_path),original);
    assert.deepEqual((await store.getTradeScreenshot(images.rows[0].id))!.imageData,original);
    const retry=await store.upsertTrade(input,key);assert.equal(retry.trade.id,first.trade.id);
    assert.equal((await pool.query('select * from trade_logs')).rowCount,1);
    assert.equal((await pool.query('select * from trade_screenshots')).rowCount,1);
    r2.controls.corrupt=true;
    await assert.rejects(store.updateTrade(first.trade.id,"branden",{...input,screenshots:[inline,inline]}),/integrity/);
    assert.deepEqual((await pool.query('select screenshots from trade_logs')).rows[0].screenshots,first.trade.screenshots);
    r2.controls.corrupt=false;
    await assert.rejects(writeTradeWithInlineScreenshots(pool,first.trade.id,"branden",[inline],0,
      'update trade_logs set screenshots=$1 where id=$2 returning id',[null,'missing']),/not found/);
    const png=await sharp({create:{width:4,height:4,channels:3,background:'#567'}}).png().toBuffer();
    const uploadKey=randomUUID();
    const a=await store.saveTradeScreenshot(first.trade.id,"branden","new.png","image/png",png,uploadKey);
    const b=await store.saveTradeScreenshot(first.trade.id,"branden","new.png","image/png",png,uploadKey);
    assert.deepEqual(a,b);
    const refs=(await pool.query('select screenshots from trade_logs')).rows[0].screenshots;
    assert.equal(refs.filter((value:string)=>value===a.url).length,1);
    const afterUploadRetry=await store.upsertTrade({...input,screenshots:[]},key);
    assert.deepEqual(afterUploadRetry.trade.screenshots,refs);
    await assert.rejects(store.saveTradeScreenshot(first.trade.id,"cam","new.png","image/png",png,uploadKey),/not found/);
  } finally {
    await pool?.end();r2.restore();process.env=previous;
    await admin.query(`drop schema ${schema} cascade`);await admin.end();
  }
});
