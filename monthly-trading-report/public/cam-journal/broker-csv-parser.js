(function registerCamBrokerCsvParser(root, factory) {
  const parser = factory();

  if (typeof module === "object" && module.exports) {
    module.exports = parser;
  }

  root.CamBrokerCsvParser = parser;
})(typeof globalThis !== "undefined" ? globalThis : this, function createCamBrokerCsvParser() {
  function parseCsv(text) {
    text = text.replace(/^\uFEFF/, "");
    const rows = [];
    let row = [];
    let cell = "";
    let inQuotes = false;

    for (let index = 0; index < text.length; index += 1) {
      const character = text[index];
      const next = text[index + 1];

      if (character === '"') {
        if (inQuotes && next === '"') {
          cell += '"';
          index += 1;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (character === "," && !inQuotes) {
        row.push(cell);
        cell = "";
      } else if ((character === "\n" || character === "\r") && !inQuotes) {
        if (character === "\r" && next === "\n") {
          index += 1;
        }
        row.push(cell);
        rows.push(row);
        row = [];
        cell = "";
      } else {
        cell += character;
      }
    }

    if (cell.length || row.length) {
      row.push(cell);
      rows.push(row);
    }

    return rows.filter((candidate) => candidate.some((value) => String(value).trim() !== ""));
  }

  function cleanNumber(value) {
    if (value === undefined || value === null) {
      return 0;
    }

    const cleaned = String(value).replace(/[$,()]/g, "").trim();
    if (!cleaned || cleaned === "--") {
      return 0;
    }

    const sign = String(value).includes("(") && String(value).includes(")") ? -1 : 1;
    const number = Number(cleaned);
    return Number.isFinite(number) ? number * sign : 0;
  }

  function normalizeDate(mmddyy) {
    const parts = String(mmddyy).split("/");
    if (parts.length !== 3) {
      return mmddyy;
    }

    const [month, day, shortYear] = parts.map((part) => part.padStart(2, "0"));
    const year = Number(shortYear) < 70 ? `20${shortYear}` : `19${shortYear}`;
    return `${year}-${month}-${day}`;
  }

  function parseTradeDescription(description) {
    const text = String(description || "").trim();
    const match = text.match(
      /^(BOT|SOLD)\s+([+-]?(?:\d+(?:,\d{3})*)(?:\.\d+)?)\s+(.+?)\s+@(\.?\d+(?:\.\d+)?)/i
    );

    if (!match) {
      return null;
    }

    const action = match[1].toUpperCase();
    const qty = Math.abs(Number(match[2].replace(/,/g, "")));
    const instrument = match[3].trim();
    const price = Number(match[4]);

    if (!Number.isFinite(qty) || qty <= 0 || !Number.isFinite(price) || price <= 0) {
      return null;
    }

    const ticker = (instrument.split(/\s+/)[0] || "").toUpperCase();
    const isOption = /\b(CALL|PUT)\b/i.test(instrument);
    const signedQty = action === "BOT" ? qty : -qty;
    return { action, qty, signedQty, instrument, ticker, price, isOption };
  }

  function normalizedText(value) {
    return String(value || "").trim().replace(/\s+/g, " ");
  }

  function normalizedRef(value) {
    return normalizedText(value).replace(/^=/, "").replace(/^"|"$/g, "");
  }

  function normalizedNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? String(number) : "";
  }

  function canonicalExecutionKey(execution) {
    return [
      normalizeDate(execution.date || ""),
      normalizedText(execution.time),
      normalizedRef(execution.ref),
      normalizedText(execution.action).toUpperCase(),
      normalizedNumber(execution.qty),
      normalizedText(execution.instrument).toUpperCase(),
      normalizedNumber(execution.price),
      normalizedNumber(execution.amount)
    ].join("|");
  }

  function normalizedExecution(execution) {
    const action = normalizedText(execution.action).toUpperCase();
    const qty = Math.abs(Number(execution.qty || 0));
    const signedQuantity = Number(execution.signedQty);
    const signedQty =
      Number.isFinite(signedQuantity) && signedQuantity !== 0
        ? signedQuantity
        : action === "BOT"
          ? qty
          : -qty;
    const normalized = {
      ...execution,
      date: normalizeDate(execution.date || ""),
      time: normalizedText(execution.time),
      ref: normalizedRef(execution.ref),
      action,
      qty,
      signedQty,
      price: Number(execution.price || 0),
      instrument: normalizedText(execution.instrument),
      ticker: normalizedText(execution.ticker || normalizedText(execution.instrument).split(/\s+/)[0]).toUpperCase(),
      amount: Number(execution.amount || 0),
      miscFees: Number(execution.miscFees || 0),
      commissions: Number(execution.commissions || 0),
      description: normalizedText(execution.description)
    };
    normalized.executionKey = canonicalExecutionKey(normalized);
    return normalized;
  }

  function statementExecutions(rows) {
    const executions = [];
    const skippedTradeRows = [];
    let header = null;

    for (const row of rows) {
      const firstCell = String(row[0] || "").trim();
      const secondCell = String(row[1] || "").trim();
      const typeCell = String(row[2] || "").trim().toUpperCase();

      if (firstCell === "Futures Statements") {
        break;
      }

      if (firstCell.toUpperCase() === "DATE" && secondCell.toUpperCase() === "TIME" && typeCell === "TYPE") {
        header = row;
        continue;
      }

      if (!header || typeCell !== "TRD") {
        continue;
      }

      const description = row[4] || "";
      const parsedDescription = parseTradeDescription(description);
      if (!parsedDescription) {
        skippedTradeRows.push({
          date: normalizeDate(row[0]),
          time: row[1] || "",
          ref: row[3] || "",
          description
        });
        continue;
      }

      executions.push(normalizedExecution({
        date: normalizeDate(row[0]),
        time: row[1] || "",
        type: row[2],
        ref: row[3] || "",
        description,
        miscFees: cleanNumber(row[5]),
        commissions: cleanNumber(row[6]),
        amount: cleanNumber(row[7]),
        balance: cleanNumber(row[8]),
        ...parsedDescription
      }));
    }

    return { executions, skippedTradeRows };
  }

  function defaultIdFactory() {
    if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
      return globalThis.crypto.randomUUID();
    }
    return `broker-import-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function numberOrBlank(value) {
    if (value === "" || value === null || value === undefined) {
      return "";
    }
    const number = Number(value);
    return Number.isFinite(number) ? number : "";
  }

  function summarizeRoundTrip(key, executions, closed = true, idFactory = defaultIdFactory) {
    const normalizedExecutions = executions.map(normalizedExecution);
    const first = normalizedExecutions[0];
    const last = normalizedExecutions[normalizedExecutions.length - 1];
    const openingSign = Math.sign(first.signedQty);
    const direction = openingSign > 0 ? "Long" : "Short";
    const buys = normalizedExecutions.filter((execution) => execution.action === "BOT");
    const sells = normalizedExecutions.filter((execution) => execution.action === "SOLD");
    const boughtQty = buys.reduce((sum, execution) => sum + execution.qty, 0);
    const soldQty = sells.reduce((sum, execution) => sum + execution.qty, 0);
    const openingExecutions = direction === "Long" ? buys : sells;
    const closingExecutions = direction === "Long" ? sells : buys;
    const openingQty = openingExecutions.reduce((sum, execution) => sum + execution.qty, 0);
    const closingQty = closingExecutions.reduce((sum, execution) => sum + execution.qty, 0);
    const entry = openingQty
      ? openingExecutions.reduce((sum, execution) => sum + execution.qty * execution.price, 0) / openingQty
      : "";
    const exit = closingQty
      ? closingExecutions.reduce((sum, execution) => sum + execution.qty * execution.price, 0) / closingQty
      : "";
    const totalCash = normalizedExecutions.reduce(
      (sum, execution) => sum + execution.amount + execution.miscFees + execution.commissions,
      0
    );
    const executionKeys = normalizedExecutions.map((execution) => execution.executionKey);
    const openingKeys = openingExecutions.map((execution) => execution.executionKey);

    return {
      id: idFactory(),
      date: first.date,
      entryTime: first.time,
      exitDate: closed ? last.date : "",
      exitTime: closed ? last.time : "",
      ticker: first.ticker,
      instrument: first.instrument,
      setup: "Other",
      grade: "",
      direction,
      entry: numberOrBlank(entry),
      exit: numberOrBlank(exit),
      stop: "",
      size: Math.max(boughtQty, soldQty),
      risk: "",
      pl: closed ? Number(totalCash.toFixed(2)) : "",
      rMultiple: "",
      portfolioTag: "",
      emotion: "",
      status: closed ? "Closed" : "Open",
      checklist: "",
      notes: "",
      source: "Broker CSV import",
      screenshots: [],
      importOpenKey: `${first.instrument}|${direction}|${openingKeys.join("~")}`,
      importTradeKey: `${first.instrument}|${executionKeys.join("~")}`,
      executionKeys,
      rawExecutions: normalizedExecutions
    };
  }

  function groupExecutionsIntoTrades(executions, options = {}) {
    const idFactory = options.idFactory || defaultIdFactory;
    const byInstrument = new Map();

    executions.map(normalizedExecution).forEach((execution) => {
      const key = execution.instrument;
      if (!byInstrument.has(key)) {
        byInstrument.set(key, []);
      }
      byInstrument.get(key).push(execution);
    });

    const imported = [];
    for (const [key, list] of byInstrument.entries()) {
      list.sort((left, right) => `${left.date}${left.time}`.localeCompare(`${right.date}${right.time}`));
      let position = 0;
      let bucket = [];

      for (const execution of list) {
        const before = position;
        position += execution.signedQty;
        bucket.push(execution);
        if (before !== 0 && position === 0) {
          imported.push(summarizeRoundTrip(key, bucket, true, idFactory));
          bucket = [];
        }
      }

      if (bucket.length) {
        imported.push(summarizeRoundTrip(key, bucket, false, idFactory));
      }
    }

    return imported.sort((left, right) =>
      `${left.date || ""}${left.entryTime || ""}`.localeCompare(`${right.date || ""}${right.entryTime || ""}`)
    );
  }

  const MANUAL_FIELDS = [
    "setup",
    "setupId",
    "setupVersion",
    "setupScore",
    "grade",
    "stop",
    "target",
    "risk",
    "stfAtr",
    "rMultiple",
    "portfolioTag",
    "secondaryTag",
    "mistakeTag",
    "emotion",
    "checklist",
    "notes",
    "screenshots",
    "playbookScreenshotIndex",
    "customTags"
  ];

  function cleanBrokerImportNoteText(text) {
    return String(text || "")
      .split(/\n+/)
      .filter((line) => !/Broker import updated this trade with new execution data\.?/i.test(line.trim()))
      .join("\n")
      .trim();
  }

  function preserveManualFields(existing, imported) {
    const merged = { ...existing, ...imported, id: existing.id || imported.id };
    for (const field of MANUAL_FIELDS) {
      if (Object.prototype.hasOwnProperty.call(existing, field)) {
        merged[field] = field === "notes" ? cleanBrokerImportNoteText(existing[field]) : existing[field];
      }
    }
    return merged;
  }

  function syntheticExecutionsFromTrade(trade) {
    if (Array.isArray(trade.rawExecutions) && trade.rawExecutions.length) {
      return trade.rawExecutions.map(normalizedExecution);
    }

    const instrument = trade.instrument || trade.ticker || trade.symbol || "";
    const ticker = trade.ticker || trade.symbol || String(instrument).split(/\s+/)[0] || "";
    const qty = Math.abs(Number(trade.size || trade.quantity || trade.shares || trade.qty || 0));
    const entry = Number(trade.entry || 0);
    const exit = Number(trade.exit || 0);
    if (!qty || !entry) {
      return [];
    }

    const isLong = String(trade.direction || "Long").toLowerCase() !== "short";
    const openAction = isLong ? "BOT" : "SOLD";
    const closeAction = isLong ? "SOLD" : "BOT";
    const executions = [
      normalizedExecution({
        date: trade.date || "",
        time: trade.entryTime || "",
        action: openAction,
        qty,
        signedQty: isLong ? qty : -qty,
        price: entry,
        instrument,
        ticker,
        amount: isLong ? -(qty * entry) : qty * entry,
        miscFees: 0,
        commissions: 0,
        description: `${openAction} ${isLong ? "+" : "-"}${qty} ${instrument} @${entry}`
      })
    ];

    if (String(trade.status || "").toLowerCase() === "closed" && exit) {
      executions.push(
        normalizedExecution({
          date: trade.exitDate || trade.date || "",
          time: trade.exitTime || "",
          action: closeAction,
          qty,
          signedQty: isLong ? -qty : qty,
          price: exit,
          instrument,
          ticker,
          amount: isLong ? qty * exit : -(qty * exit),
          miscFees: 0,
          commissions: 0,
          description: `${closeAction} ${isLong ? "-" : "+"}${qty} ${instrument} @${exit}`
        })
      );
    }

    return executions;
  }

  function dedupeExecutions(executions) {
    const seen = new Set();
    return (executions || [])
      .map(normalizedExecution)
      .filter((execution) => {
        if (seen.has(execution.executionKey)) {
          return false;
        }
        seen.add(execution.executionKey);
        return true;
      })
      .sort((left, right) => `${left.date}${left.time}`.localeCompare(`${right.date}${right.time}`));
  }

  function netExecutionPosition(executions) {
    return (executions || []).reduce((sum, execution) => sum + normalizedExecution(execution).signedQty, 0);
  }

  function executionKeysForTrade(trade) {
    if (Array.isArray(trade.rawExecutions) && trade.rawExecutions.length) {
      return new Set(trade.rawExecutions.map((execution) => normalizedExecution(execution).executionKey));
    }
    return new Set();
  }

  function normalizedLifecycleValue(value) {
    return normalizedText(value).toUpperCase();
  }

  function lifecycleMatches(existing, imported) {
    const existingInstrument = normalizedLifecycleValue(existing.instrument || existing.ticker || existing.symbol);
    const importedInstrument = normalizedLifecycleValue(imported.instrument || imported.ticker || imported.symbol);
    const sameNumber = (left, right, tolerance = 0.000001) => {
      const leftNumber = Number(left);
      const rightNumber = Number(right);
      return Number.isFinite(leftNumber) && Number.isFinite(rightNumber) && Math.abs(leftNumber - rightNumber) <= tolerance;
    };

    return (
      existingInstrument === importedInstrument &&
      normalizedLifecycleValue(existing.direction) === normalizedLifecycleValue(imported.direction) &&
      normalizeDate(existing.date || "") === normalizeDate(imported.date || "") &&
      normalizedText(existing.entryTime) === normalizedText(imported.entryTime) &&
      normalizeDate(existing.exitDate || "") === normalizeDate(imported.exitDate || "") &&
      normalizedText(existing.exitTime) === normalizedText(imported.exitTime) &&
      normalizedLifecycleValue(existing.status) === normalizedLifecycleValue(imported.status) &&
      sameNumber(existing.size || existing.quantity || existing.shares || existing.qty, imported.size) &&
      sameNumber(existing.entry, imported.entry, 0.005) &&
      (normalizedLifecycleValue(imported.status) !== "CLOSED" || sameNumber(existing.exit, imported.exit, 0.005))
    );
  }

  function upsertImportedTrades(existingTrades, importedTrades, options = {}) {
    const idFactory = options.idFactory || defaultIdFactory;
    const trades = existingTrades.map((trade) => ({ ...trade }));
    const result = { added: 0, updated: 0, ignored: 0, closedOpen: 0, ambiguous: 0 };

    function rebuildExistingWithImported(existing, imported) {
      const combined = dedupeExecutions([
        ...syntheticExecutionsFromTrade(existing),
        ...(imported.rawExecutions || [])
      ]);
      const netPosition = netExecutionPosition(combined);
      const rebuilt = summarizeRoundTrip(
        existing.instrument || imported.instrument || existing.ticker || imported.ticker,
        combined,
        netPosition === 0,
        idFactory
      );
      const merged = preserveManualFields(existing, rebuilt);
      if (String(existing.status || "").toLowerCase() === "open" && String(merged.status || "").toLowerCase() === "closed") {
        result.closedOpen += 1;
      }
      return merged;
    }

    for (const importedTrade of importedTrades) {
      const imported = { ...importedTrade };
      const importedKeys = new Set(
        (imported.rawExecutions || []).map((execution) => normalizedExecution(execution).executionKey)
      );
      imported.executionKeys = [...importedKeys];
      imported.importTradeKey = `${imported.instrument}|${imported.executionKeys.join("~")}`;

      let index = trades.findIndex((trade) => {
        const existingKeys = executionKeysForTrade(trade);
        if (!existingKeys.size || !importedKeys.size) {
          return false;
        }
        return [...importedKeys].some((key) => existingKeys.has(key));
      });

      if (index === -1) {
        const lifecycleCandidates = trades
          .map((trade, candidateIndex) => ({ trade, candidateIndex }))
          .filter(({ trade }) => lifecycleMatches(trade, imported));

        if (lifecycleCandidates.length === 1) {
          index = lifecycleCandidates[0].candidateIndex;
        } else if (lifecycleCandidates.length > 1) {
          result.ambiguous += 1;
          continue;
        }
      }

      if (index === -1 && normalizedLifecycleValue(imported.status) === "OPEN") {
        const openCandidates = trades
          .map((trade, candidateIndex) => ({ trade, candidateIndex }))
          .filter(({ trade }) => {
            const sameInstrument =
              normalizedLifecycleValue(trade.instrument || trade.ticker || trade.symbol) ===
              normalizedLifecycleValue(imported.instrument || imported.ticker);
            return (
              normalizedLifecycleValue(trade.status) === "OPEN" &&
              sameInstrument &&
              netExecutionPosition(syntheticExecutionsFromTrade(trade)) !== 0
            );
          });

        if (openCandidates.length === 1) {
          index = openCandidates[0].candidateIndex;
        } else if (openCandidates.length > 1) {
          result.ambiguous += 1;
          continue;
        }
      }

      if (index === -1) {
        trades.push(imported);
        result.added += 1;
        continue;
      }

      const existing = trades[index];
      const existingKeys = executionKeysForTrade(existing);
      const overlap = [...importedKeys].filter((key) => existingKeys.has(key)).length;
      if (importedKeys.size && overlap === importedKeys.size && existingKeys.size >= importedKeys.size) {
        result.ignored += 1;
        continue;
      }

      const before = JSON.stringify(existing);
      trades[index] = overlap > 0
        ? rebuildExistingWithImported(existing, imported)
        : preserveManualFields(existing, imported);
      if (before === JSON.stringify(trades[index])) {
        result.ignored += 1;
      } else {
        result.updated += 1;
      }
    }

    return { trades, result };
  }

  return {
    parseCsv,
    cleanNumber,
    normalizeDate,
    parseTradeDescription,
    canonicalExecutionKey,
    normalizedExecution,
    statementExecutions,
    groupExecutionsIntoTrades,
    preserveManualFields,
    lifecycleMatches,
    upsertImportedTrades,
    syntheticExecutionsFromTrade,
    dedupeExecutions
  };
});
