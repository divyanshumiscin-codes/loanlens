const form = document.getElementById("form");
const details = document.getElementById("result").querySelector("#details");
const gauge = document.getElementById("gauge");
const arc = document.getElementById("arc");
const pctText = document.getElementById("pct");
const rows = document.getElementById("rows");
const btn = document.getElementById("go");
const inr = n => "₹" + Math.round(n).toLocaleString("en-IN");
const BANDS = {likely: ["Likely approved", "yes"], borderline: ["Borderline", "mid"], unlikely: ["Unlikely", "no"]};
const PRESETS = {
  strong: {Married: "Yes", Dependents: "0", Education: "Graduate", Self_Employed: "No", ApplicantIncome: 8000, CoapplicantIncome: 3000, Credit_History: "1", Property_Area: "Semiurban", LoanAmount: 120, Loan_Amount_Term: 360},
  average: {Married: "Yes", Dependents: "1", Education: "Graduate", Self_Employed: "No", ApplicantIncome: 4000, CoapplicantIncome: 1500, Credit_History: "1", Property_Area: "Urban", LoanAmount: 150, Loan_Amount_Term: 360},
  weak: {Married: "No", Dependents: "3+", Education: "Not Graduate", Self_Employed: "Yes", ApplicantIncome: 2500, CoapplicantIncome: 0, Credit_History: "0", Property_Area: "Rural", LoanAmount: 200, Loan_Amount_Term: 240}
};

// Small helper: make an element with text (textContent keeps it safe from injection)
function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}

// Same EMI formula as the backend, so the live preview matches the final answer
const emi = (p, n) => { const r = 9 / 1200; return p * r * Math.pow(1 + r, n) / (Math.pow(1 + r, n) - 1); };

function syncLoan() {
  const amount = +form.LoanAmount.value, term = +form.Loan_Amount_Term.value;
  document.getElementById("amtOut").textContent = inr(amount * 1000);
  document.getElementById("termOut").textContent = term + " months (" + +(term / 12).toFixed(1) + " years)";
  document.getElementById("emiLive").textContent = inr(emi(amount * 1000, term));
}
form.LoanAmount.addEventListener("input", syncLoan);
form.Loan_Amount_Term.addEventListener("input", syncLoan);
syncLoan();

// CIBIL is optional: the slider is only sent when the box is ticked
const useCibil = document.getElementById("useCibil");
function syncCibil() {
  form.Cibil_Score.disabled = !useCibil.checked;
  document.getElementById("cibilOut").textContent = useCibil.checked ? form.Cibil_Score.value : "not used";
}
useCibil.addEventListener("change", syncCibil);
form.Cibil_Score.addEventListener("input", syncCibil);
syncCibil();

// The meter under the result: drag to see the chance at other scores (no server call needed)
function cibilMeter(c) {
  const zoneOf = s => s >= 750 ? "good" : s >= 650 ? "borderline" : "weak";
  const NAMES = {weak: "weak zone", borderline: "borderline zone", good: "good zone"};
  const box = el("div", "", "cibil");
  const bar = el("div", "", "cbar");
  ["weak", "borderline", "good"].forEach(z => bar.append(el("span", "", "z-" + z)));
  const pin = el("i", "", "pin");
  bar.append(pin);
  const slider = el("input");
  slider.type = "range"; slider.min = 300; slider.max = 900; slider.step = 10;
  slider.value = c.used ? c.score : 750;
  const read = el("p", "", "cread");
  const update = () => {
    const s = +slider.value, z = zoneOf(s);
    pin.style.left = ((s - 300) / 6) + "%";
    read.textContent = "CIBIL " + s + " (" + NAMES[z] + "): chance about " + Math.round(c.chances[z] * 100) + "%";
  };
  slider.addEventListener("input", update);
  update();
  box.append(el("h3", "CIBIL score meter"), bar, slider, read,
    el("p", "The zones are an assumption and lenders differ. The model has no CIBIL data: a good score is treated as good credit history, a weak score as poor credit history, and borderline as the average of the two.", "note"));
  return box;
}

function showResult(d) {
  const pct = Math.round(d.probability * 100);
  const [word, cls] = BANDS[d.band];
  gauge.setAttribute("class", cls);
  arc.setAttribute("stroke-dasharray", Math.max(pct, 0.01) + " 100"); // CSS animates this
  pctText.textContent = pct + "%";

  const facts = el("dl");
  [["Estimated EMI", inr(d.emi) + " / month"], ["EMI as share of income", d.emi_share.toFixed(0) + "%"]]
    .forEach(([k, v]) => facts.append(el("dt", k), el("dd", v)));
  const why = el("ul");
  d.reasons.forEach(r => why.append(el("li", r)));

  const printBtn = el("button", "Print result", "ghost");
  printBtn.type = "button";
  printBtn.addEventListener("click", () => window.print());
  const moved = el("ul", "", "drivers");
  d.drivers.forEach(x => {
    const li = el("li", x.label);
    li.append(el("b", (x.effect > 0 ? "+" : "") + x.effect + " points", x.effect > 0 ? "up" : "down"));
    moved.append(li);
  });
  details.replaceChildren(
    el("div", word, "stamp " + cls),
    ...d.warnings.map(w => el("p", w, "warn")),
    facts, el("h3", "Why this result"), why,
    ...(d.drivers.length ? [el("h3", "What moved the score"), el("p", "Compared with a typical applicant in the training data.", "note"), moved] : []),
    cibilMeter(d.cibil),
    printBtn
  );
}

async function loadHistory() {
  const items = await (await fetch("/api/history")).json();
  rows.replaceChildren(...items.map(i => {
    const tr = el("tr");
    [i.created_at, inr(i.ApplicantIncome + i.CoapplicantIncome), inr(i.LoanAmount * 1000),
     Math.round(i.probability * 100) + "%"].forEach(v => tr.append(el("td", v)));
    const td = el("td");
    td.append(el("span", i.decision, "pill " + (i.decision === "Approved" ? "yes" : "no")));
    tr.append(td);
    return tr;
  }));
}

form.addEventListener("submit", async e => {
  e.preventDefault();
  btn.disabled = true; btn.textContent = "Checking...";
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(Object.fromEntries(new FormData(form)))
    });
    const data = await res.json();
    if (!res.ok) { details.replaceChildren(el("p", Object.values(data.errors).join(" "), "error")); return; }
    showResult(data);
    loadHistory();
  } catch (err) {
    details.replaceChildren(el("p", "Could not reach the server. Is app.py still running?", "error"));
  } finally {
    btn.disabled = false; btn.textContent = "Check eligibility";
  }
});

loadHistory();

document.getElementById("clearBtn").addEventListener("click", async () => {
  if (!confirm("Delete all saved checks?")) return;
  await fetch("/api/history", {method: "DELETE"});
  loadHistory();
});

// Example profiles: fill the form (they use the credit history choice, not CIBIL) and run the check
document.querySelectorAll("[data-preset]").forEach(b => b.addEventListener("click", () => {
  Object.entries(PRESETS[b.dataset.preset]).forEach(([k, v]) => { form.elements[k].value = v; });
  useCibil.checked = false;
  syncCibil();
  syncLoan();
  form.requestSubmit();
}));