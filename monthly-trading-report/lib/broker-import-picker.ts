/** A single portfolio chooser shared by the journal's broker import entry points. */
export async function chooseBrokerImportPortfolio(defaultValue = "", known: string[] = []): Promise<string | null> {
  let saved = known;
  try {
    const response = await fetch("/api/settings/branden-portfolios", { cache: "no-store" });
    if (response.ok) { const data = await response.json(); saved = [...known, ...(data.portfolios || [])]; }
  } catch { /* The caller's portfolios and IBKR Personal remain available offline. */ }
  return new Promise(resolve => {
    const dialog = document.createElement("dialog");
    dialog.className = "broker-import-dialog";
    dialog.setAttribute("aria-labelledby", "broker-import-title");
    const form = document.createElement("form"); form.method = "dialog";
    const title = document.createElement("h2"); title.id = "broker-import-title"; title.textContent = "Import broker statement";
    const hint = document.createElement("p"); hint.textContent = "Choose a portfolio, then select a CF PDF or IBKR Flex XML file.";
    const label = document.createElement("label"); label.textContent = "Portfolio";
    const select = document.createElement("select"); select.name = "portfolio";
    for (const name of [...new Set([...saved, defaultValue, "IBKR Personal"].filter(Boolean))].sort()) {
      const option = document.createElement("option"); option.value = name; option.textContent = name; select.append(option);
    }
    select.value = defaultValue || "IBKR Personal";
    label.append(select);
    const actions = document.createElement("div"); actions.className = "broker-import-actions";
    const cancel = document.createElement("button"); cancel.type = "button"; cancel.textContent = "Cancel"; cancel.onclick = () => dialog.close();
    const submit = document.createElement("button"); submit.type = "submit"; submit.textContent = "Choose file";
    let result: string | null = null;
    form.onsubmit = event => { event.preventDefault(); const chosen = select.value.trim(); if (!chosen) return; result = chosen; dialog.close(); };
    dialog.onclose = () => { dialog.remove(); resolve(result); };
    actions.append(cancel, submit); form.append(title, hint, label, actions); dialog.append(form); document.body.append(dialog); dialog.showModal();
  });
}
