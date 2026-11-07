import type { TradeLogEntry } from "./types";

/** Gross cost of remaining FIFO lots, expressed against one budget. */
export function buildPositionSizeHistory(trades: TradeLogEntry[], start: string, end: string, budget: number) {
  if (!start || !end || start > end || !Number.isFinite(budget) || budget <= 0) return { series: [], points: [] };
  const selected = trades.filter(t => !t.hidden && t.entryDate <= end && (!t.exitDate || t.exitDate >= start));
  const series = selected.map((trade, index) => ({ key: `position${index}`, label: `${trade.symbol} · ${trade.entryDate}`, trade }));
  const dates: string[] = [];
  const cursor = new Date(`${start}T00:00:00Z`), last = new Date(`${end}T00:00:00Z`);
  if (!Number.isFinite(cursor.getTime()) || !Number.isFinite(last.getTime())) return { series: [], points: [] };
  while (cursor <= last) { dates.push(`${cursor.toISOString().slice(0,10)} 23:59:59`); cursor.setUTCDate(cursor.getUTCDate()+1); }
  const events = series.map(({trade}) => {
    if (trade.executions?.length) return [...trade.executions].sort((a,b) => `${a.date}T${a.time}`.localeCompare(`${b.date}T${b.time}`));
    return [
      {type: "ENTRY", date: trade.entryDate, time:trade.openTime || "00:00:00", shares: trade.shares, price: trade.avgEntry},
      ...(trade.status !== "OPEN" && trade.exitDate ? [{type: "EXIT", date: trade.exitDate, time:trade.closeTime || "00:00:00", shares: trade.shares, price: trade.exitPrice}] : [])
    ];
  });
  const stamp = (e: {date:string; time?:string}) => `${e.date} ${e.time || "00:00:00"}`;
  const timestamps = [...new Set([`${start} 00:00:00`, ...dates, ...events.flatMap(history=>history.filter(e=>e.date>=start && e.date<=end).map(stamp))])].sort();
  const lots = series.map(() => [] as {shares:number; price:number}[]);
  const offsets = series.map(() => 0);
  const points = timestamps.map(date => {
    const row: Record<string, number | string> = {date, total:0, dollars:0};
    series.forEach(({key}, i) => {
      const history = events[i];
      while (offsets[i] < history.length && stamp(history[offsets[i]]) <= date) {
        const e=history[offsets[i]++];
        if (e.type === "ENTRY") lots[i].push({shares:Math.max(0,e.shares),price:Math.abs(e.price)});
        else {
          let remaining=Math.max(0,e.shares);
          while (remaining>1e-9 && lots[i].length) {
            const lot=lots[i][0], quantity=Math.min(remaining,lot.shares);
            lot.shares-=quantity; remaining-=quantity;
            if(lot.shares<1e-9) lots[i].shift();
          }
        }
      }
      const dollars=lots[i].reduce((sum,l)=>sum+l.shares*l.price,0);
      row[key]=dollars/budget*100;
      row.total=Number(row.total)+Number(row[key]); row.dollars=Number(row.dollars)+dollars;
    });
    return row;
  });
  return {series,points};
}
