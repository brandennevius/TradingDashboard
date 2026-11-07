"use client";
import { useMemo, useState } from "react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, ReferenceLine } from "recharts";
import type { TradeLogEntry } from "@/lib/types";
import { buildPositionSizeHistory } from "@/lib/position-size-history";

const colors=["#4f7045","#8c6a4a","#6d8f9b","#b29958","#9a7090","#8caa73","#ad7565","#638f87"];
const money=(n:number)=>new Intl.NumberFormat("en-US",{style:"currency",currency:"USD",maximumFractionDigits:0}).format(n);
export default function PositionSizeChart({trades,portfolio,startDate,endDate,suggestedBudget}: {trades:TradeLogEntry[];portfolio:string;startDate:string;endDate:string;suggestedBudget?:number}) {
  const key=`position-size-budget:${portfolio || "all"}`;
  const [budgetText,setBudgetText]=useState(()=>{try{return localStorage.getItem(key) ?? (suggestedBudget && suggestedBudget>0 ? String(suggestedBudget):"");}catch{return "";}});
  const budget=Number(budgetText);
  const data=useMemo(()=>buildPositionSizeHistory(trades.filter(t=>!portfolio || t.portfolioTag===portfolio),startDate,endDate,budget),[trades,portfolio,startDate,endDate,budget]);
  const latest=data.points.at(-1), peak=data.points.reduce((max,p)=>Math.max(max,Number(p.total)),0);
  return <article className="trade-chart-panel top-chart position-size-panel">
    <div className="trade-chart-heading"><div><p className="eyebrow">Position sizing</p><h3>Position size / drawdown</h3><p className="position-size-description">{portfolio || "All portfolios"} · Remaining shares at entry cost, stacked by trade. One drawdown amount applies to the entire date range.</p></div>
      <label className="position-size-budget">Drawdown budget ($)<input type="number" min="0.01" step="any" placeholder="e.g. 15000" value={budgetText} onChange={e=>{setBudgetText(e.target.value);try{localStorage.setItem(key,e.target.value);}catch{/* Optional browser preference. */}}}/></label>
    </div>
    {budget>0 && Number.isFinite(budget) ? <>
      <div className="position-size-summary"><span>At range end <strong>{Number(latest?.total || 0).toFixed(2)}%</strong></span><span>Position value <strong>{money(Number(latest?.dollars || 0))}</strong></span><span>Peak exposure <strong>{peak.toFixed(2)}%</strong></span></div>
      {data.series.length ? <ResponsiveContainer width="100%" height={340}><AreaChart data={data.points} margin={{top:18,right:18,bottom:8,left:8}}>
        <CartesianGrid strokeDasharray="3 5" vertical={false} stroke="rgba(47,53,45,.14)"/>
        <XAxis dataKey="date" tickFormatter={d=>String(d).slice(5,10)} minTickGap={35} tick={{fill:"#6f7469",fontSize:11}}/>
        <YAxis domain={[0,"auto"]} tickFormatter={v=>`${Number(v).toFixed(0)}%`} width={65} tick={{fill:"#6f7469",fontSize:11}}/>
        <ReferenceLine y={100} stroke="#a85757" strokeDasharray="4 4" ifOverflow="extendDomain"/>
        <Tooltip content={({active,payload,label})=>active && payload?.length ? <div className="position-size-tooltip"><strong>{String(label)}</strong>{payload.map(p=><div key={String(p.dataKey)} style={{color:p.color}}>{p.name}: {Number(p.value).toFixed(2)}% · {money(Number(p.value)*budget/100)}</div>)}<strong>Total: {Number(payload[0]?.payload.total).toFixed(2)}% · {money(Number(payload[0]?.payload.dollars))}</strong></div>:null}/>
        {data.series.map((s,i)=><Area key={s.key} dataKey={s.key} name={s.label} stackId="positions" type="stepAfter" stroke={colors[i%colors.length]} fill={colors[i%colors.length]} fillOpacity={.55} strokeWidth={2} isAnimationActive={false}/>)}
      </AreaChart></ResponsiveContainer>:<p className="status">No positions held during this date range.</p>}
      <div className="position-size-legend">{data.series.map((s,i)=><span key={s.key}><i style={{background:colors[i%colors.length]}}/>{s.label}</span>)}</div>
      <small>Gross position exposure, including shorts. Excludes commissions and price changes; this is not potential loss at your stops. Budget is remembered in this browser.</small>
    </>:<p className="status">Enter your drawdown budget to plot position sizes. For example, $1,500 held against $15,000 is 10%.</p>}
  </article>;
}
