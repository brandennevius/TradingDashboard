"use client";

import { useEffect, useRef, useState } from "react";
import {
  CandlestickSeries,
  ColorType,
  LineSeries,
  createChart,
  type CandlestickData,
  type IChartApi,
  type LineData,
  type Time
} from "lightweight-charts";

type SpyCandle = CandlestickData<Time> & {
  time: Time;
};

function movingAverageData(candles: SpyCandle[], period: number, type: "ema" | "sma"): LineData<Time>[] {
  const items: LineData<Time>[] = [];

  if (!candles.length || period <= 0) {
    return items;
  }

  if (type === "sma") {
    for (let index = period - 1; index < candles.length; index += 1) {
      const slice = candles.slice(index - period + 1, index + 1);
      const value = slice.reduce((total, candle) => total + candle.close, 0) / period;
      items.push({ time: candles[index].time, value });
    }

    return items;
  }

  const multiplier = 2 / (period + 1);
  let ema = 0;

  candles.forEach((candle, index) => {
    if (index < period - 1) {
      return;
    }

    if (index === period - 1) {
      const seed = candles.slice(0, period);
      ema = seed.reduce((total, item) => total + item.close, 0) / period;
    } else {
      ema = (candle.close - ema) * multiplier + ema;
    }

    items.push({ time: candle.time, value: ema });
  });

  return items;
}

function initialVisibleRange(candles: SpyCandle[]) {
  const lastCandle = candles[candles.length - 1];
  const lastDate = new Date(`${String(lastCandle.time)}T00:00:00Z`);

  if (Number.isNaN(lastDate.getTime())) {
    return null;
  }

  const rangeStartDate = new Date(lastDate);
  rangeStartDate.setUTCMonth(rangeStartDate.getUTCMonth() - 2);
  const rangeStart = rangeStartDate.toISOString().slice(0, 10);
  const firstVisibleCandle = candles.find((candle) => String(candle.time) >= rangeStart) || candles[0];

  return {
    from: firstVisibleCandle.time,
    to: lastCandle.time
  };
}

export default function SpyMarketCycleChart() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [candles, setCandles] = useState<SpyCandle[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadCandles() {
      const response = await fetch("/api/market-data/spy", { cache: "no-store" });
      const data = await response.json();

      if (cancelled) {
        return;
      }

      if (!response.ok) {
        setError(data.error || "Could not load SPY candles.");
        setCandles([]);
        return;
      }

      setError("");
      setCandles(data.candles || []);
    }

    loadCandles();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!containerRef.current || !candles.length) {
      return;
    }

    const container = containerRef.current;
    const chart = createChart(container, {
      autoSize: true,
      height: 390,
      layout: {
        background: { type: ColorType.Solid, color: "#fffaf0" },
        textColor: "#6f7469"
      },
      grid: {
        horzLines: { color: "rgba(47, 53, 45, 0.12)" },
        vertLines: { color: "rgba(47, 53, 45, 0.08)" }
      },
      rightPriceScale: {
        borderColor: "rgba(47, 53, 45, 0.14)"
      },
      timeScale: {
        borderColor: "rgba(47, 53, 45, 0.14)",
        timeVisible: true
      },
      crosshair: {
        mode: 1
      }
    });
    const series = chart.addSeries(CandlestickSeries, {
      upColor: "#4f7045",
      borderUpColor: "#4f7045",
      wickUpColor: "#4f7045",
      downColor: "#a65a55",
      borderDownColor: "#a65a55",
      wickDownColor: "#a65a55"
    });

    series.setData(candles);

    const ema8 = chart.addSeries(LineSeries, {
      color: "#f3a6bb",
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
      title: "8 EMA"
    });
    ema8.setData(movingAverageData(candles, 8, "ema"));

    const ema21 = chart.addSeries(LineSeries, {
      color: "#3f7fce",
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
      title: "21 EMA"
    });
    ema21.setData(movingAverageData(candles, 21, "ema"));

    const sma50 = chart.addSeries(LineSeries, {
      color: "#d45a5a",
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
      title: "50 SMA"
    });
    sma50.setData(movingAverageData(candles, 50, "sma"));

    const visibleRange = initialVisibleRange(candles);

    if (visibleRange) {
      chart.timeScale().setVisibleRange(visibleRange);
    } else {
      chart.timeScale().fitContent();
    }

    chartRef.current = chart;

    return () => {
      chart.remove();
      chartRef.current = null;
    };
  }, [candles]);

  return (
    <div className="spy-market-chart">
      {error ? <p className="muted">{error}</p> : null}
      {!error && !candles.length ? <p className="muted">Loading SPY candles...</p> : null}
      <div className="spy-market-chart-legend" aria-label="SPY moving averages">
        <span>
          <i style={{ backgroundColor: "#f3a6bb" }} />
          8 EMA
        </span>
        <span>
          <i style={{ backgroundColor: "#3f7fce" }} />
          21 EMA
        </span>
        <span>
          <i style={{ backgroundColor: "#d45a5a" }} />
          50 SMA
        </span>
      </div>
      <div ref={containerRef} className="spy-market-chart-canvas" />
    </div>
  );
}
