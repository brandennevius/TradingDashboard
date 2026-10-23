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

  function rFromPrice(price, trade) {
    const inputs = normalizedInputs(trade || {});
    const normalizedPrice = finiteNumber(price);
    if (!inputs || !(normalizedPrice > 0)) return null;
    const move = inputs.direction === 'short'
      ? inputs.entry - normalizedPrice
      : normalizedPrice - inputs.entry;
    return move * inputs.size / inputs.risk;
  }

  function priceFromR(rValue, trade) {
    const inputs = normalizedInputs(trade || {});
    const normalizedR = finiteNumber(rValue);
    if (!inputs || normalizedR === null) return null;
    const move = normalizedR * inputs.risk / inputs.size;
    return inputs.direction === 'short' ? inputs.entry - move : inputs.entry + move;
  }

  return { finiteNumber, rFromPrice, priceFromR };
});
