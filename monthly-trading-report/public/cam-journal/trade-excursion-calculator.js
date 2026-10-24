(function attachTradeExcursionCalculator(root, factory) {
  const calculator = factory();
  if (typeof module === 'object' && module.exports) module.exports = calculator;
  else root.CamTradeExcursion = calculator;
})(typeof globalThis !== 'undefined' ? globalThis : this, function createTradeExcursionCalculator() {
  function finiteNumber(value) {
    if (value === null || value === undefined || String(value).trim() === '') return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function normalizedInputs({ entry, size, risk, direction }) {
    const normalizedEntry = finiteNumber(entry);
    const normalizedSize = Math.abs(finiteNumber(size));
    const normalizedRisk = Math.abs(finiteNumber(risk));
    if (!(normalizedEntry > 0) || !(normalizedSize > 0) || !(normalizedRisk > 0)) return null;
    return {
      entry: normalizedEntry,
      size: normalizedSize,
      risk: normalizedRisk,
      direction: String(direction || 'Long').toLowerCase() === 'short' ? 'short' : 'long'
    };
  }

  function normalizedRiskReference({ entry, size, risk, riskDistance, direction }) {
    const inputs = normalizedInputs({ entry, size, risk, direction });
    const distance = Math.abs(finiteNumber(riskDistance));
    if (!(finiteNumber(entry) > 0) || (!(distance > 0) && !inputs)) return null;
    return {
      entry: finiteNumber(entry),
      distance: distance > 0 ? distance : inputs.risk / inputs.size,
      direction: String(direction || 'Long').toLowerCase() === 'short' ? 'short' : 'long'
    };
  }

  function executionTimestamp(execution) {
    const rawDate = String(execution?.date || '').trim();
    const dateMatch = rawDate.match(/^(\d{1,2})\/(\d{1,2})\/(\d{2}|\d{4})$/);
    const date = dateMatch
      ? `${dateMatch[3].length === 2 ? (Number(dateMatch[3]) < 70 ? '20' : '19') + dateMatch[3] : dateMatch[3]}-${dateMatch[1].padStart(2, '0')}-${dateMatch[2].padStart(2, '0')}`
      : rawDate;
    const timeMatch = String(execution?.time || '').trim().match(/^(\d{1,2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?\s*(AM|PM)?$/i);
    if (!date || !timeMatch) return NaN;
    let hour = Number(timeMatch[1]);
    const suffix = String(timeMatch[4] || '').toUpperCase();
    if (suffix === 'PM' && hour < 12) hour += 12;
    if (suffix === 'AM' && hour === 12) hour = 0;
    return new Date(`${date}T${String(hour).padStart(2, '0')}:${timeMatch[2]}:${timeMatch[3] || '00'}`).getTime();
  }

  function initialEntry(executions, trade = {}, windowMs = 30000) {
    const direction = String(trade.direction || 'Long').toLowerCase() === 'short' ? 'short' : 'long';
    const openingAction = direction === 'short' ? 'SOLD' : 'BOT';
    const openings = (executions || [])
      .filter(execution => String(execution?.action || '').toUpperCase() === openingAction && finiteNumber(execution?.qty) > 0 && finiteNumber(execution?.price) > 0)
      .slice()
      .sort((left, right) => {
        const leftMs = executionTimestamp(left);
        const rightMs = executionTimestamp(right);
        if (Number.isFinite(leftMs) && Number.isFinite(rightMs)) return leftMs - rightMs;
        return `${left.date || ''}${left.time || ''}`.localeCompare(`${right.date || ''}${right.time || ''}`);
      });
    if (!openings.length) {
      const entry = finiteNumber(trade.entry);
      const size = Math.abs(finiteNumber(trade.size));
      return entry > 0 && size > 0 ? { entry, size, fillCount: 1, source: 'trade entry' } : null;
    }
    const firstMs = executionTimestamp(openings[0]);
    const initialFills = openings.filter((execution, index) => {
      if (index === 0) return true;
      const timestamp = executionTimestamp(execution);
      return Number.isFinite(firstMs) && Number.isFinite(timestamp) && timestamp - firstMs <= windowMs;
    });
    const size = initialFills.reduce((total, execution) => total + Math.abs(finiteNumber(execution.qty) || 0), 0);
    if (!(size > 0)) return null;
    const entry = initialFills.reduce((total, execution) => total + Math.abs(finiteNumber(execution.qty) || 0) * finiteNumber(execution.price), 0) / size;
    return {
      entry,
      size,
      fillCount: initialFills.length,
      source: initialFills.length > 1 ? `${initialFills.length} fills within 30 sec` : 'first fill'
    };
  }

  function rFromPrice(price, trade) {
    const inputs = normalizedRiskReference(trade || {});
    const normalizedPrice = finiteNumber(price);
    if (!inputs || !(normalizedPrice > 0)) return null;
    const move = inputs.direction === 'short'
      ? inputs.entry - normalizedPrice
      : normalizedPrice - inputs.entry;
    return move / inputs.distance;
  }

  function priceFromR(rValue, trade) {
    const inputs = normalizedRiskReference(trade || {});
    const normalizedR = finiteNumber(rValue);
    if (!inputs || normalizedR === null) return null;
    const move = normalizedR * inputs.distance;
    return inputs.direction === 'short' ? inputs.entry - move : inputs.entry + move;
  }

  return { finiteNumber, executionTimestamp, initialEntry, rFromPrice, priceFromR };
});
