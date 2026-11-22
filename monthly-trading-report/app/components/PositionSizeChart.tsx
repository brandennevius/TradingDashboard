"use client";
import { useMemo, useState } from "react";
import { Area, ComposedChart, Line, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, ReferenceLine } from "recharts";
import type { TradeLogEntry } from "@/lib/types";
import { buildPositionSizeHistory } from "@/lib/position-size-history";
import { buildAccountEquityHistory, type AccountEquitySnapshot } from "@/lib/account-equity-history";

const noEquity: AccountEquitySnapshot[] = [];
const noPortfolios: string[] = [];

const colors=["#4f7045","#8c6a4a","#6d8f9b","#b29958","#9a7090","#8caa73","#ad7565","#638f87"];
const money=(n:number)=>new Intl.NumberFormat("en-US",{style:"currency",currency:"USD",maximumFractionDigits:0}).format(n);
export default function PositionSizeChart({trades,portfolio,startDate,endDate,suggestedBudget,equityHistory=noEquity,equityPortfolios=noPortfolios,equityError=""}: {trades:TradeLogEntry[];portfolio:string;startDate:string;endDate:string;suggestedBudget?:number;equityHistory?:AccountEquitySnapshot[];equityPortfolios?:string[];equityError?:string}) {
  const key=`position-size-budget:${portfolio || "all"}`;
  const [budgetText,setBudgetText]=useState(()=>{try{return localStorage.getItem(key) ?? (suggestedBudget && suggestedBudget>0 ? String(suggestedBudget):"");}catch{return "";}});
  const budget=Number(budgetText);
  const data=useMemo(()=>buildPositionSizeHistory(trades.filter(t=>!portfolio || t.portfolioTag===portfolio),startDate,endDate,budget),[trades,portfolio,startDate,endDate,budget]);
  const equity = useMemo(() => buildAccountEquityHistory(equityHistory, equityPortfolios, portfolio, startDate, endDate), [equityHistory, equityPortfolios, portfolio, startDate, endDate]);
  const chartPoints = useMemo(() => {
    const recorded = new Map(equity.map(point => [point.date, point.accountEquity]));
    return data.points.map(point => ({ ...point, accountEquity: recorded.get(String(point.date)) ?? null }));
  }, [data.points, equity]);
  const latest=data.points.at(-1), peak=data.points.reduce((max,p)=>Math.max(max,Number(p.total)),0);
  return <article className="trade-chart-panel top-chart position-size-panel">
    <div className="trade-chart-heading"><div><p className="eyebrow">Position sizing</p><h3>Position size / drawdown</h3><p className="position-size-description">{portfolio || "All portfolios"} · Remaining shares at entry cost, stacked by trade. One drawdown amount applies to the entire date range.</p></div>
      <label className="position-size-budget">Drawdown budget ($)<input type="number" min="0.01" step="any" placeholder="e.g. 15000" value={budgetText} onChange={e=>{setBudgetText(e.target.value);try{localStorage.setItem(key,e.target.value);}catch{/* Optional browser preference. */}}}/></label>
    </div>
    {budget>0 && Number.isFinite(budget) ? <>
      <div className="position-size-summary"><span>At range end <strong>{Number(latest?.total || 0).toFixed(2)}%</strong></span><span>Position value <strong>{money(Number(latest?.dollars || 0))}</strong></span><span>Peak exposure <strong>{peak.toFixed(2)}%</strong></span></div>
      {data.series.length || equity.length ? <ResponsiveContainer width="100%" height={340}><ComposedChart data={chartPoints} margin={{top:18,right:18,bottom:8,left:8}}>
        <CartesianGrid strokeDasharray="3 5" vertical={false} stroke="rgba(47,53,45,.14)"/>
        <XAxis dataKey="date" tickFormatter={d=>String(d).slice(5,10)} minTickGap={35} tick={{fill:"#6f7469",fontSize:11}}/>
        <YAxis yAxisId="exposure" domain={[0,"auto"]} tickFormatter={v=>`${Number(v).toFixed(0)}%`} width={65} tick={{fill:"#6f7469",fontSize:11}}/>
        {equity.length > 0 ? <YAxis yAxisId="equity" orientation="right" domain={["auto", "auto"]} tickFormatter={money} width={95} tick={{fill:"#111111",fontSize:11}} /> : null}
        <ReferenceLine yAxisId="exposure" y={100} stroke="#a85757" strokeDasharray="4 4" ifOverflow="extendDomain"/>
        <Tooltip content={({active,payload,label}) => {
          if (!active || !payload?.length) return null;
          // Every series appears in Recharts' payload, even before entry or after exit.
          // Use the hovered point's remaining exposure, not the trade's current status.
          const positions = payload.filter(p => p.dataKey !== "accountEquity" && Number.isFinite(Number(p.value)) && Number(p.value) > 0);
          return <div className="position-size-tooltip">
            <strong>{String(label)}</strong>
            {positions.length ? positions.map(p => <div key={String(p.dataKey)} style={{color:p.color}}>{p.name}: {Number(p.value).toFixed(2)}% · {money(Number(p.value)*budget/100)}</div>) : <div>No open positions at this time.</div>}
            <strong>Total: {Number(payload[0]?.payload.total).toFixed(2)}% · {money(Number(payload[0]?.payload.dollars))}</strong>
            {payload[0]?.payload.accountEquity != null ? <strong style={{color:"#111111"}}>Account equity: {money(Number(payload[0].payload.accountEquity))}</strong> : null}
          </div>;
        }}/>

        {data.series.map((s,i)=><Area yAxisId="exposure" key={s.key} dataKey={s.key} name={s.label} stackId="positions" type="stepAfter" stroke={colors[i%colors.length]} fill={colors[i%colors.length]} fillOpacity={.55} strokeWidth={2} isAnimationActive={false}/>)}
        {equity.length > 0 ? <Line yAxisId="equity" dataKey="accountEquity" name="Account equity" type="linear" stroke="#111111" strokeWidth={2.5} dot={{r:3,fill:"#111111"}} activeDot={{r:5}} connectNulls isAnimationActive={false} /> : null}
      </ComposedChart></ResponsiveContainer>:<p className="status">No positions held during this date range.</p>}
      <div className="position-size-legend">{equity.length > 0 ? <span><i style={{background:"#111111",height:3}}/>Account equity ($, right axis)</span> : null}{data.series.map((s,i)=><span key={s.key}><i style={{background:colors[i%colors.length]}}/>{s.label}</span>)}</div>
      <p className="position-size-description">{equityError || (equity.length ? "Black line: recorded account equity; connects available statement dates. Hover a recorded date for its dollar value." : portfolio ? "No recorded account equity for this portfolio in the selected period." : "No complete same-date account equity for all portfolios in this period. Select a portfolio to see its available history.")}</p>
      <small>Gross position exposure, including shorts. Excludes commissions and price changes; this is not potential loss at your stops. Budget is remembered in this browser.</small>
    </>:<p className="status">Enter your drawdown budget to plot position sizes. For example, $1,500 held against $15,000 is 10%.</p>}
  </article>;
}
