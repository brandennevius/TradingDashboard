import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { parseIbkrFlex, mergeIbkrData, buildIbkrTrades } from "../lib/ibkr-flex";

// Synthetic data; real account exports remain outside the repository.
const sample = `<FlexQueryResponse type="AF"><FlexStatements><FlexStatement accountId="TEST" fromDate="20261019" toDate="20261027">
<AccountInformation accountId="TEST" currency="USD"/>
<ChangeInNAV startingValue="0" endingValue="99"/>
<EquitySummaryInBase><EquitySummaryByReportDateInBase reportDate="20261027" cash="49" stock="50" total="99"/></EquitySummaryInBase>
<OpenPositions><OpenPosition accountId="TEST" symbol="TEST" currency="USD" assetCategory="STK" position="0.5" positionValue="50" markPrice="100" costBasisPrice="102" fifoPnlUnrealized="-1"/></OpenPositions>
<Trades><Trade accountId="TEST" symbol="TEST" currency="USD" assetCategory="STK" conid="1" ibExecID="fill1" transactionID="trade1" levelOfDetail="EXECUTION" openCloseIndicator="O" buySell="BUY" quantity="0.5" tradePrice="102" ibCommission="0" fifoPnlRealized="0" tradeDate="20261027" dateTime="20261027;094232"/></Trades>
<CashTransactions><CashTransaction accountId="TEST" transactionID="deposit1" levelOfDetail="DETAIL" type="Deposits/Withdrawals" amount="100" currency="USD" dateTime="20261019" settleDate="20261019" reportDate="20261023" availableForTradingDate="20261023"/></CashTransactions>
<SecuritiesInfo><SecurityInfo conid="1" symbol="TEST" currency="USD"/></SecuritiesInfo></FlexStatement></FlexStatements></FlexQueryResponse>`;

test("fractional holding, cash dates, identifiers, duplicate and overlap handling", () => {
  const parsed = parseIbkrFlex(sample);
  const merged = mergeIbkrData(parsed, parsed);
  assert.equal(merged.executions.length,1);
  assert.equal(merged.cashTransactions.length,1);
  assert.equal(merged.cashTransactions[0].reportDate,"20261023");
  assert.equal(merged.cashTransactions[0].dateTime,"20261019");
  const [t] = buildIbkrTrades(merged,"branden","IBKR Personal",[]);
  assert.equal(t.shares,.5); assert.equal(t.avgEntry,102); assert.equal(t.pnl,0);
  assert.equal(t.executions![0].sourceKey,"TEST:fill1");
  const saved = { ...t, id:"saved",createdAt:"",updatedAt:"",hidden:false,groupId:"",groupRole:"none", executions:t.executions!, stopPrice:90,notes:"Keep my review" } as const;
  const [again] = buildIbkrTrades(merged,"branden","IBKR Personal",[saved]);
  assert.equal(again.stopPrice,90); assert.equal(again.notes,"Keep my review");
  const cleared = buildIbkrTrades(merged,"branden","IBKR Personal",[{...saved,stopPrice:0}]);
  assert.equal(cleared[0].stopPrice,0);
});

test("reject malformed XML, entities, wrong accounts, missing identifiers and incomplete history", () => {
  assert.throws(()=>parseIbkrFlex('<broken>'),/valid XML/);
  assert.throws(()=>parseIbkrFlex('<!DOCTYPE x>'+sample),/entity/);
  assert.throws(()=>parseIbkrFlex(sample.replace('ibExecID="fill1"','')),/Execution ID/);
  const parsed = parseIbkrFlex(sample);
  assert.throws(()=>mergeIbkrData(parsed,{...parsed,accountId:"OTHER"}),/different IBKR/);
  assert.throws(()=>buildIbkrTrades({...parsed,executions:[]},"branden","IBKR Personal",[]),/Missing execution/);
});

test("daily reports preserve earlier cash history and older uploads do not regress holdings",()=>{
  const history=parseIbkrFlex(sample);
  const daily={...history,fromDate:"2026-10-27",cashTransactions:[]};
  assert.equal(mergeIbkrData(history,daily).cashTransactions.length,1);
  const old={...history,toDate:"2026-10-23",positions:[],executions:[]};
  assert.equal(mergeIbkrData(daily,old).toDate,"2026-10-27");
  assert.equal(mergeIbkrData(daily,old).positions.length,1);
});

test("store keeps IBKR isolated, preserves stops, and saves cash ledger and dated equity", async()=>{
  const cwd=process.cwd(), dir=await fs.mkdtemp(path.join(os.tmpdir(),"ibkr-store-"));
  const db=process.env.DATABASE_URL; delete process.env.DATABASE_URL; process.chdir(dir);
  try {
    const store=await import('../lib/store');
    const parsed=parseIbkrFlex(sample), hash="a".repeat(64);
    await store.importIbkrFlexStatement("branden","IBKR Personal",parsed,hash,"test.xml");
    let [trade]=await store.listTrades();
    await store.updateTrade(trade.id,"branden",{...trade,stopPrice:90,notes:"manual review"});
    await store.importIbkrFlexStatement("branden","IBKR Personal",parsed,hash,"test.xml");
    const list=await store.listTrades(); assert.equal(list.length,1); assert.equal(list[0].stopPrice,90); assert.equal(list[0].notes,"manual review");
    assert.equal(list[0].id,trade.id);
    const snapshots=await store.listBrokerPortfolioSnapshots("branden","IBKR Personal");
    assert.equal(snapshots[0].source,"IBKR_FLEX_XML"); assert.equal(snapshots[0].currentEquity,99);
    const settings=JSON.parse(await fs.readFile(path.join(dir,'data/app-settings.json'),'utf8'));
    assert.equal(settings['ibkr_flex:branden:IBKR Personal'].cashTransactions.length,1);
    await assert.rejects(store.importIbkrFlexStatement("branden","IBKR Personal",{...parsed,accountId:"OTHER"},hash,"test.xml"),/different IBKR/);
    assert.equal((await store.listTrades())[0].stopPrice,90);
  } finally { process.chdir(cwd); if(db!==undefined)process.env.DATABASE_URL=db; await fs.rm(dir,{recursive:true,force:true}); }
});

test("provided daily and historical exports reconcile and share one execution", {skip:!process.env.IBKR_HISTORY_FILE}, async()=>{
  const history=parseIbkrFlex(await fs.readFile(process.env.IBKR_HISTORY_FILE!,"utf8"));
  const daily=parseIbkrFlex(await fs.readFile(process.env.IBKR_DAILY_FILE!,"utf8"));
  const state=mergeIbkrData(history,daily);
  assert.equal(state.executions.length,1); assert.equal(state.cashTransactions.length,2);
  assert.equal(state.cashTransactions.reduce((s,r)=>s+Number(r.amount),0),250);
  const [t]=buildIbkrTrades(state,"branden","IBKR Personal",[]);
  assert.equal(t.symbol,"LNG"); assert.equal(t.shares,.8); assert.equal(t.executions!.length,1);
  assert.equal(Number(state.nav.endingValue),248.342057135);
});
