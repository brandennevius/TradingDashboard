"use client";

import { type MouseEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  CandlestickSeries,
  ColorType,
  LineStyle,
  createChart,
  createSeriesMarkers,
  type CandlestickData,
  type IChartApi,
  type SeriesMarker,
  type Time
} from "lightweight-charts";
import type { TradeSide } from "@/lib/types";

type TradePriceChartProps = {
  symbol: string;
  side: TradeSide;
  entryDate: string;
  exitDate: string;
  avgEntry: number;
  exitPrice: number;
  stopPrice: number;
  takeProfitPrice: number;
};
type ChartTimeframe = "1h" | "4h" | "1d" | "1wk" | "1mo";
type Candle = CandlestickData<Time> & { time: Time };
type DrawingMode = "trendline" | "horizontal" | "callout" | null;
type DrawingPoint = { x: number; y: number };
type ChartDrawing =
  | { id: string; type: "trendline"; start: DrawingPoint; end: DrawingPoint }
  | { id: string; type: "horizontal"; y: number }
  | { id: string; type: "callout"; point: DrawingPoint; text: string };

const timeframeOptions: { value: ChartTimeframe; label: string }[] = [
  { value: "1h", label: "1H" },
  { value: "4h", label: "4H" },
  { value: "1d", label: "1D" },
  { value: "1wk", label: "1W" },
  { value: "1mo", label: "1M" }
];

function cleanSymbol(value: string) {
  return value.trim().replace(/^#/, "").toUpperCase();
}

function moneyLabel(price: number) {
  return `$${price.toFixed(price >= 10 ? 2 : 4)}`;
}

function isValidPrice(price: number) {
  return Number.isFinite(price) && price > 0;
}

function candleDay(value: string) {
  return value.slice(0, 10);
}

function nearestCandleTime(targetDate: string, candleTimes: string[]) {
  if (!targetDate || !candleTimes.length) {
    return "";
  }

  const exact = candleTimes.find((time) => candleDay(time) === targetDate);
  if (exact) {
    return exact;
  }

  return candleTimes.find((time) => candleDay(time) > targetDate) || [...candleTimes].reverse().find((time) => candleDay(time) < targetDate) || "";
}

function buildMarkers(props: TradePriceChartProps, candleTimes: string[]): SeriesMarker<Time>[] {
  const entryTime = nearestCandleTime(props.entryDate, candleTimes);
  const exitTime = nearestCandleTime(props.exitDate, candleTimes);
  const markers: SeriesMarker<Time>[] = [];
  const isShort = props.side === "SHORT";

  if (entryTime && isValidPrice(props.avgEntry)) {
    markers.push({
      time: entryTime as Time,
      position: isShort ? "aboveBar" : "belowBar",
      shape: isShort ? "arrowDown" : "arrowUp",
      color: isShort ? "#a65a55" : "#4f7045",
      text: `${isShort ? "Sell short" : "Buy"} ${moneyLabel(props.avgEntry)}`
    });
  }

  if (exitTime && isValidPrice(props.exitPrice)) {
    markers.push({
      time: exitTime as Time,
      position: isShort ? "belowBar" : "aboveBar",
      shape: isShort ? "arrowUp" : "arrowDown",
      color: "#8c6a4a",
      text: `${isShort ? "Cover" : "Sell"} ${moneyLabel(props.exitPrice)}`
    });
  }

  return markers;
}

export default function TradePriceChart(props: TradePriceChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const fullscreenContainerRef = useRef<HTMLDivElement | null>(null);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [error, setError] = useState("");
  const [timeframe, setTimeframe] = useState<ChartTimeframe>("1d");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [drawingMode, setDrawingMode] = useState<DrawingMode>(null);
  const [pendingTrendlinePoint, setPendingTrendlinePoint] = useState<DrawingPoint | null>(null);
  const [drawings, setDrawings] = useState<ChartDrawing[]>([]);
  const symbol = cleanSymbol(props.symbol);

  useEffect(() => {
    let cancelled = false;

    async function loadCandles() {
      if (!symbol) {
        setCandles([]);
        setError("Add a symbol to load the chart.");
        return;
      }

      const response = await fetch(`/api/market-data/${encodeURIComponent(symbol)}?timeframe=${timeframe}`, { cache: "no-store" });
      const data = await response.json();

      if (cancelled) {
        return;
      }

      if (!response.ok) {
        setError(data.error || `Could not load ${symbol} candles.`);
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
  }, [symbol, timeframe]);

  const candleTimes = useMemo(() => candles.map((candle) => String(candle.time)), [candles]);
  const markers = useMemo(() => buildMarkers(props, candleTimes), [props, candleTimes]);

  function renderChart(container: HTMLDivElement, height: number): IChartApi | null {
    if (!candles.length) {
      return null;
    }

    const chart = createChart(container, {
      autoSize: true,
      height,
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
        timeVisible: timeframe === "1h" || timeframe === "4h"
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

    if (isValidPrice(props.avgEntry)) {
      series.createPriceLine({
        price: props.avgEntry,
        color: "#4f7045",
        lineWidth: 2,
        lineStyle: LineStyle.Solid,
        axisLabelVisible: true,
        title: `Entry ${moneyLabel(props.avgEntry)}`
      });
    }

    if (isValidPrice(props.exitPrice)) {
      series.createPriceLine({
        price: props.exitPrice,
        color: "#8c6a4a",
        lineWidth: 2,
        lineStyle: LineStyle.Solid,
        axisLabelVisible: true,
        title: `Exit ${moneyLabel(props.exitPrice)}`
      });
    }

    if (isValidPrice(props.stopPrice)) {
      series.createPriceLine({
        price: props.stopPrice,
        color: "#a65a55",
        lineWidth: 2,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: `Stop ${moneyLabel(props.stopPrice)}`
      });
    }

    if (isValidPrice(props.takeProfitPrice)) {
      series.createPriceLine({
        price: props.takeProfitPrice,
        color: "#6f8f5f",
        lineWidth: 2,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: `Target ${moneyLabel(props.takeProfitPrice)}`
      });
    }

    createSeriesMarkers(series, markers);
    chart.timeScale().fitContent();
    return chart;
  }

  useEffect(() => {
    if (!containerRef.current || !candles.length) {
      return;
    }

    const chart = renderChart(containerRef.current, 420);

    return () => {
      chart?.remove();
    };
  }, [candles, markers, props.avgEntry, props.exitPrice, props.stopPrice, props.takeProfitPrice, timeframe]);

  useEffect(() => {
    if (!isFullscreen || !fullscreenContainerRef.current || !candles.length) {
      return;
    }

    const chart = renderChart(fullscreenContainerRef.current, Math.max(520, Math.floor(window.innerHeight * 0.72)));

    return () => {
      chart?.remove();
    };
  }, [isFullscreen, candles, markers, props.avgEntry, props.exitPrice, props.stopPrice, props.takeProfitPrice, timeframe]);

  function drawingPointFromEvent(event: MouseEvent<HTMLElement>) {
    const bounds = event.currentTarget.getBoundingClientRect();
    return {
      x: ((event.clientX - bounds.left) / bounds.width) * 100,
      y: ((event.clientY - bounds.top) / bounds.height) * 100
    };
  }

  function handleDrawingSurfaceClick(event: MouseEvent<HTMLElement>) {
    if (!drawingMode) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();

    const point = drawingPointFromEvent(event);
    const id = `drawing-${Date.now()}-${Math.random().toString(16).slice(2)}`;

    if (drawingMode === "horizontal") {
      setDrawings((current) => [...current, { id, type: "horizontal", y: point.y }]);
      return;
    }

    if (drawingMode === "trendline") {
      if (!pendingTrendlinePoint) {
        setPendingTrendlinePoint(point);
        return;
      }

      setDrawings((current) => [...current, { id, type: "trendline", start: pendingTrendlinePoint, end: point }]);
      setPendingTrendlinePoint(null);
      return;
    }

    const text = window.prompt("Callout text");
    if (text?.trim()) {
      setDrawings((current) => [...current, { id, type: "callout", point, text: text.trim() }]);
    }
  }

  function chooseDrawingMode(nextMode: Exclude<DrawingMode, null>) {
    setDrawingMode((current) => (current === nextMode ? null : nextMode));
    setPendingTrendlinePoint(null);
  }

  function renderDrawingOverlay(isFullscreenOverlay = false) {
    return (
      <button
        aria-label={drawingMode ? `Place ${drawingMode} drawing` : "Chart drawings"}
        className={drawingMode ? "chart-drawing-layer active" : "chart-drawing-layer"}
        type="button"
        onClick={handleDrawingSurfaceClick}
      >
        <svg preserveAspectRatio="none" viewBox="0 0 100 100">
          {drawings.map((drawing) => {
            if (drawing.type === "horizontal") {
              return <line key={drawing.id} className="chart-drawing-line horizontal" x1="0" x2="100" y1={drawing.y} y2={drawing.y} />;
            }

            if (drawing.type === "trendline") {
              return (
                <line
                  key={drawing.id}
                  className="chart-drawing-line trendline"
                  x1={drawing.start.x}
                  x2={drawing.end.x}
                  y1={drawing.start.y}
                  y2={drawing.end.y}
                />
              );
            }

            return (
              <g key={drawing.id} className="chart-callout">
                <line x1={drawing.point.x} x2={Math.min(96, drawing.point.x + 6)} y1={drawing.point.y} y2={Math.max(7, drawing.point.y - 7)} />
                <rect x={Math.min(72, drawing.point.x + 6)} y={Math.max(3, drawing.point.y - 12)} width="26" height="9" rx="2" />
                <text x={Math.min(74, drawing.point.x + 8)} y={Math.max(9, drawing.point.y - 6)}>
                  {drawing.text.slice(0, isFullscreenOverlay ? 36 : 24)}
                </text>
              </g>
            );
          })}
          {pendingTrendlinePoint ? <circle className="chart-pending-point" cx={pendingTrendlinePoint.x} cy={pendingTrendlinePoint.y} r="1.2" /> : null}
        </svg>
      </button>
    );
  }

  return (
    <div className="trade-price-chart">
      <div className="trade-price-chart-toolbar">
        <div className="trade-timeframe-toggle" aria-label="Chart timeframe">
          {timeframeOptions.map((option) => (
            <button key={option.value} className={timeframe === option.value ? "active" : ""} type="button" onClick={() => setTimeframe(option.value)}>
              {option.label}
            </button>
          ))}
        </div>
        <div className="chart-tool-actions">
          <button className={drawingMode === "trendline" ? "active" : ""} type="button" onClick={() => chooseDrawingMode("trendline")}>
            Trendline
          </button>
          <button className={drawingMode === "horizontal" ? "active" : ""} type="button" onClick={() => chooseDrawingMode("horizontal")}>
            Horizontal
          </button>
          <button className={drawingMode === "callout" ? "active" : ""} type="button" onClick={() => chooseDrawingMode("callout")}>
            Callout
          </button>
          <button type="button" onClick={() => setDrawings([])}>
            Clear
          </button>
          <button className="trade-muted-button chart-maximize-button" type="button" onClick={() => setIsFullscreen(true)}>
            Maximize
          </button>
        </div>
      </div>
      {error ? <p className="muted">{error}</p> : null}
      {!error && !candles.length ? <p className="muted">Loading {symbol} candles...</p> : null}
      <div className="trade-price-chart-surface">
        <div ref={containerRef} className="trade-price-chart-canvas" />
        {renderDrawingOverlay()}
      </div>
      {isFullscreen ? (
        <div className="chart-fullscreen-backdrop" role="dialog" aria-modal="true" aria-label={`${symbol} full screen chart`}>
          <section className="chart-fullscreen-panel">
            <div className="chart-fullscreen-header">
              <div>
                <p className="eyebrow">{symbol}</p>
                <h3>Execution chart</h3>
              </div>
              <div className="trade-price-chart-toolbar">
                <div className="trade-timeframe-toggle" aria-label="Full screen chart timeframe">
                  {timeframeOptions.map((option) => (
                    <button key={option.value} className={timeframe === option.value ? "active" : ""} type="button" onClick={() => setTimeframe(option.value)}>
                      {option.label}
                    </button>
                  ))}
                </div>
                <div className="chart-tool-actions">
                  <button className={drawingMode === "trendline" ? "active" : ""} type="button" onClick={() => chooseDrawingMode("trendline")}>
                    Trendline
                  </button>
                  <button className={drawingMode === "horizontal" ? "active" : ""} type="button" onClick={() => chooseDrawingMode("horizontal")}>
                    Horizontal
                  </button>
                  <button className={drawingMode === "callout" ? "active" : ""} type="button" onClick={() => chooseDrawingMode("callout")}>
                    Callout
                  </button>
                  <button type="button" onClick={() => setDrawings([])}>
                    Clear
                  </button>
                </div>
                <button className="trade-muted-button" type="button" onClick={() => setIsFullscreen(false)}>
                  Close
                </button>
              </div>
            </div>
            {error ? <p className="muted">{error}</p> : null}
            <div className="trade-price-chart-surface">
              <div ref={fullscreenContainerRef} className="trade-price-chart-canvas fullscreen" />
              {renderDrawingOverlay(true)}
            </div>
          </section>
        </div>
      ) : null}
    </div>
  );
}
