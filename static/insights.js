const $ = id => document.getElementById(id);
const pct = v => (v * 100).toFixed(1) + "%";

function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}

// One horizontal bar row: label, bar (fraction 0..1), value text, optional tag
function barRow(label, fraction, text, tag, best) {
  const row = el("div", "", "barrow" + (best ? " best" : ""));
  const track = el("div", "", "track2"), fill = el("span");
  fill.style.width = Math.max(fraction * 100, 1) + "%";
  track.append(fill);
  row.append(el("span", label), track, el("strong", text), el("span", tag || "", tag ? "tag" : ""));
  return row;
}

async function load() {
  const data = await (await fetch("/api/insights")).json();
  const m = data.metrics;
  if (!m) {
    $("stats").replaceChildren(el("p", "No model report yet. Run python train_model.py, then refresh.", "error"));
    return;
  }
  $("testRows").textContent = m.test_rows;

  $("stats").replaceChildren(...[
    [pct(m.accuracy), "Accuracy", "Share of all decisions that were right."],
    [pct(m.precision), "Precision", "When it says approved, how often it is right."],
    [pct(m.recall), "Recall", "Of the truly good applicants, how many it approves."],
    [pct(m.f1), "F1 score", "One number that balances precision and recall."]
  ].map(([n, t, note]) => { const d = el("div", "", "stat"); d.append(el("strong", n), el("span", t), el("small", note)); return d; }));

  $("models").replaceChildren(...m.models.map(r =>
    barRow(r.name, r.cv, pct(r.cv) + " CV", r.name === m.best ? "Chosen" : "", r.name === m.best)));
  $("models").append(el("p", "Test score of the chosen model: " + pct(m.accuracy), "lead"));

  const [[rr, rw], [aw, ar]] = m.confusion;
  const cell = (n, text, cls) => { const d = el("div", "", cls); d.append(el("b", n), document.createTextNode(text)); return d; };
  $("matrix").replaceChildren(
    el("div", "", "h"), el("div", "Model said: Rejected", "h"), el("div", "Model said: Approved", "h"),
    el("div", "Actually rejected", "h"), cell(rr, "correctly rejected", "good"), cell(rw, "wrongly approved (risky)", "bad"),
    el("div", "Actually approved", "h"), cell(aw, "wrongly rejected", "bad"), cell(ar, "correctly approved", "good"));

  const top = Math.max(...m.importance.map(i => i.value));
  $("factors").replaceChildren(...m.importance.map(i => barRow(i.feature, i.value / top, i.value + "%")));

  const eda = m.eda;
  if (eda) {
    $("edaRows").textContent = eda.rows + " past applications, " + pct(eda.approval_rate) + " approved.";
    const parts = [];
    Object.entries(eda.tables).forEach(([title, items]) => {
      parts.push(el("h3", "Approval rate by " + title.toLowerCase(), "sub"));
      items.forEach(i => parts.push(barRow(i.label + " (" + i.n + ")", i.rate, Math.round(i.rate * 100) + "%")));
    });
    parts.push(el("h3", "Applicant income (monthly)", "sub"));
    const top = Math.max(...eda.income.map(i => i.n));
    eda.income.forEach(i => parts.push(barRow(i.label, i.n / top, i.n + " people")));
    $("eda").replaceChildren(...parts);
  }

  const total = data.usage.reduce((s, u) => s + u.n, 0);
  $("usage").replaceChildren(total
    ? el("div", "", "")
    : el("p", "No checks saved yet. Make a prediction on the main page.", "empty"));
  if (total) $("usage").replaceChildren(...data.usage.map(u => barRow(u.decision, u.n / total, u.n + (u.n === 1 ? " check" : " checks"), "", u.decision === "Approved")));
}
load();
