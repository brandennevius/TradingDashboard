import test from "node:test";
import assert from "node:assert/strict";
import {buildPositionSizeHistory} from "../lib/position-size-history";
import type {TradeLogEntry,TradeExecution} from "../lib/types";
const fill=(date:string,type:"ENTRY"|"EXIT",shares:number,price:number,time="10:00:00")=>({date,time,type,shares,price} as TradeExecution);
const trade=(id:string,executions:TradeExecution[],extra:Partial<TradeLogEntry>={})=>({id,symbol:id,entryDate:executions[0].date,status:"OPEN",executions,...extra} as TradeLogEntry);
test("1500 and 500 positions stack to 13.333 percent of 15000",()=>{
 const r=buildPositionSizeHistory([trade("A",[fill("2026-11-02","ENTRY",15,100)]),trade("B",[fill("2026-11-03","ENTRY",5,100)])],"2026-11-01","2026-11-04",15000);
 const row=r.points.at(-1)!;assert.equal(row.position0,10);assert.equal(row.position1,500/15000*100);assert.ok(Math.abs(Number(row.total)-13.333333333)<1e-6);assert.equal(row.dollars,2000);
});
test("carry-in, adds, FIFO partial exits and full exits change bands",()=>{
 const t=trade("A",[fill("2026-10-30","ENTRY",10,100),fill("2026-11-02","ENTRY",10,200),fill("2026-11-03","EXIT",15,250),fill("2026-11-04","EXIT",5,250)]);
 const r=buildPositionSizeHistory([t],"2026-11-01","2026-11-05",10000);
 assert.equal(r.points[0].dollars,1000);assert.equal(r.points.find(p=>p.date==="2026-11-02 23:59:59")!.dollars,3000);assert.equal(r.points.find(p=>p.date==="2026-11-03 23:59:59")!.dollars,1000);assert.equal(r.points.at(-1)!.total,0);
});
test("intraday exposure is visible even when position closes the same day",()=>{
 const r=buildPositionSizeHistory([trade("A",[fill("2026-11-02","ENTRY",.5,300,"09:00:00"),fill("2026-11-02","EXIT",.5,310,"15:00:00")],{status:"WIN",exitDate:"2026-11-02"})],"2026-11-02","2026-11-02",1500);
 assert.equal(r.points.find(p=>p.date==="2026-11-02 09:00:00")!.total,10);assert.equal(r.points.at(-1)!.total,0);
});
test("shorts are gross exposure, hidden trades excluded, zero budgets produce no percentages",()=>{
 const trades=[trade("A",[fill("2026-11-02","ENTRY",5,100)],{side:"SHORT"}),trade("B",[fill("2026-11-02","ENTRY",5,100)],{hidden:true})];
 assert.equal(buildPositionSizeHistory(trades,"2026-11-02","2026-11-02",1000).points.at(-1)!.total,50);assert.equal(buildPositionSizeHistory(trades,"2026-11-02","2026-11-02",0).points.length,0);
});
