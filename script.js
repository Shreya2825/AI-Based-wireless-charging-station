/* =============================================================================
   AI Smart EV Charging Station — script.js
   Handles: live data, charts, gauges, Leaflet map, prediction APIs, chatbot
============================================================================= */

const API = "http://localhost:5000";

// ── Loading screen ────────────────────────────────────────────────────────────
window.addEventListener("load", () => {
  setTimeout(() => {
    const ol = document.getElementById("loadingOverlay");
    if (ol) { ol.style.opacity = "0"; ol.style.transition = "opacity 0.6s"; setTimeout(() => ol.remove(), 600); }
  }, 1900);
  initAll();
});

// ── Clock ─────────────────────────────────────────────────────────────────────
function updateClock() {
  const el = document.getElementById("live-clock");
  if (el) el.textContent = new Date().toLocaleTimeString("en-IN", { hour12: false });
}
setInterval(updateClock, 1000);
updateClock();

// ── Gauge helper (Chart.js doughnut) ─────────────────────────────────────────
function makeGauge(canvasId, value, max, color) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: "doughnut",
    data: {
      datasets: [{
        data: [value, max - value],
        backgroundColor: [color, "rgba(255,255,255,0.05)"],
        borderWidth: 0,
        circumference: 240,
        rotation: 240
      }]
    },
    options: {
      responsive: false,
      cutout: "78%",
      plugins: { legend: { display: false }, tooltip: { enabled: false } }
    }
  });
}

function updateGauge(chart, value, max) {
  if (!chart) return;
  chart.data.datasets[0].data = [Math.min(value, max), Math.max(max - value, 0)];
  chart.update("none");
}

// ── Line Chart helper ─────────────────────────────────────────────────────────
function makeLineChart(canvasId, label, color, labels) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: "line",
    data: {
      labels: labels || Array.from({ length: 12 }, (_, i) => `${i * 2}:00`),
      datasets: [{
        label,
        data: Array(12).fill(0),
        borderColor: color,
        backgroundColor: color.replace(")", ",0.08)").replace("rgb", "rgba"),
        fill: true,
        tension: 0.4,
        pointRadius: 3,
        pointBackgroundColor: color,
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: { legend: { labels: { color: "#8899b8", font: { size: 11 } } } },
      scales: {
        x: { grid: { color: "rgba(255,255,255,0.04)" }, ticks: { color: "#8899b8", font: { size: 10 } } },
        y: { grid: { color: "rgba(255,255,255,0.04)" }, ticks: { color: "#8899b8", font: { size: 10 } } }
      }
    }
  });
}

// ── LSTM dual-line chart ──────────────────────────────────────────────────────
function makeLSTMChart() {
  const ctx = document.getElementById("chart-lstm");
  if (!ctx) return null;
  const labels = Array.from({ length: 20 }, (_, i) => `T+${i}`);
  const actual    = labels.map(() => 4000 + Math.random() * 3000);
  const predicted = actual.map(v => v + (Math.random() - 0.5) * 400);
  return new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Actual Demand", data: actual, borderColor: "#00d4ff", backgroundColor: "rgba(0,212,255,0.06)", fill: true, tension: 0.4, borderWidth: 2, pointRadius: 2 },
        { label: "LSTM Predicted", data: predicted, borderColor: "#00ff88", backgroundColor: "rgba(0,255,136,0.04)", fill: true, tension: 0.4, borderWidth: 2, borderDash: [5, 3], pointRadius: 2 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: true,
      plugins: { legend: { labels: { color: "#8899b8", font: { size: 11 } } } },
      scales: {
        x: { grid: { color: "rgba(255,255,255,0.04)" }, ticks: { color: "#8899b8", font: { size: 10 } } },
        y: { grid: { color: "rgba(255,255,255,0.04)" }, ticks: { color: "#8899b8", font: { size: 10 } } }
      }
    }
  });
}

// ── Battery Health Distribution Chart ────────────────────────────────────────
function makeHealthDistChart() {
  const ctx = document.getElementById("chart-health-dist");
  if (!ctx) return null;
  return new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"],
      datasets: [{
        label: "Vehicles",
        data: [120, 340, 890, 2100, 3800, 4500],
        backgroundColor: [
          "rgba(255,59,92,0.7)", "rgba(255,107,53,0.7)", "rgba(255,214,10,0.7)",
          "rgba(0,212,255,0.7)", "rgba(0,255,136,0.7)", "rgba(0,255,136,0.9)"
        ],
        borderRadius: 6,
        borderWidth: 0
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { color: "#8899b8", font: { size: 10 } } },
        y: { grid: { color: "rgba(255,255,255,0.04)" }, ticks: { color: "#8899b8", font: { size: 10 } } }
      }
    }
  });
}

// ── Leaflet Map ───────────────────────────────────────────────────────────────
let map = null;
let stationMarkers = [];

function initMap() {
  map = L.map("map-container", { zoomControl: true }).setView([28.61, 77.21], 12);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution: "© CartoDB",
    maxZoom: 19
  }).addTo(map);
}

function addStationMarkers(stations) {
  stationMarkers.forEach(m => map.removeLayer(m));
  stationMarkers = [];
  const colors = { Online: "#00ff88", Busy: "#ffd60a", Maintenance: "#ff3b5c" };
  stations.forEach(s => {
    const col = colors[s.status] || "#00d4ff";
    const icon = L.divIcon({
      className: "",
      html: `<div style="width:18px;height:18px;background:${col};border-radius:50%;border:2px solid #fff;box-shadow:0 0 12px ${col};"></div>`,
      iconSize: [18, 18],
      iconAnchor: [9, 9]
    });
    const m = L.marker([s.lat, s.lon], { icon })
      .bindPopup(`<div style="font-family:Inter,sans-serif;color:#0a1628">
        <strong>${s.station_id}</strong><br>
        Status: <b style="color:${col}">${s.status}</b><br>
        Load: ${s.avg_load_kw} kW<br>
        Price: ₹${s.avg_price}/kWh<br>
        Sessions: ${s.sessions}
      </div>`)
      .addTo(map);
    stationMarkers.push(m);
  });
}

// ── Populate Station Table ────────────────────────────────────────────────────
function populateTable(stations) {
  const tbody = document.getElementById("station-tbody");
  if (!tbody) return;
  tbody.innerHTML = stations.map(s => {
    const cls = s.status === "Online" ? "online" : s.status === "Busy" ? "busy" : "maintenance";
    return `<tr>
      <td><strong>${s.station_id}</strong></td>
      <td>${s.lat.toFixed(4)}, ${s.lon.toFixed(4)}</td>
      <td>${s.avg_load_kw}</td>
      <td>₹${s.avg_price}</td>
      <td>${s.sessions.toLocaleString()}</td>
      <td><span class="status-pill ${cls}">${s.status}</span></td>
    </tr>`;
  }).join("");
}

// ── Fetch stations on load ────────────────────────────────────────────────────
async function fetchStations() {
  try {
    const res = await fetch(`${API}/stations`);
    const data = await res.json();
    if (data.stations) {
      addStationMarkers(data.stations);
      populateTable(data.stations);
    }
  } catch (e) {
    // Use fallback synthetic data when Flask is offline
    const fallback = Array.from({ length: 10 }, (_, i) => ({
      station_id: `STN_${String(i + 1).padStart(2, "0")}`,
      lat: 28.60 + (Math.random() - 0.5) * 0.3,
      lon: 77.20 + (Math.random() - 0.5) * 0.3,
      avg_load_kw: +(Math.random() * 3000 + 500).toFixed(1),
      avg_price: +(Math.random() * 8 + 5).toFixed(2),
      sessions: Math.floor(Math.random() * 4000 + 200),
      status: ["Online", "Online", "Busy", "Maintenance"][Math.floor(Math.random() * 4)]
    }));
    addStationMarkers(fallback);
    populateTable(fallback);
  }
}

// ── Live Data Poll ─────────────────────────────────────────────────────────────
let gaugeLoad, gaugeHealth, gaugePrice, gaugeThermal;
let chartLoad, chartPrice;
const loadHistory  = Array(12).fill(3000);
const priceHistory = Array(12).fill(8);

async function fetchLiveData() {
  let d;
  try {
    const res = await fetch(`${API}/live_data`);
    d = await res.json();
  } catch {
    // Simulate when backend offline
    const hr = new Date().getHours();
    const isPeak = (hr >= 7 && hr <= 9) || (hr >= 17 && hr <= 20);
    d = {
      current_load_kw: +(isPeak ? 5000 : 2500) + Math.random() * 600 - 300,
      grid_demand_kw: +(isPeak ? 8000 : 4000) + Math.random() * 500,
      price_per_kwh: +(isPeak ? 12 : 6.5) + Math.random() * 1.5 - 0.75,
      active_sessions: Math.floor(Math.random() * 12) + 3,
      stations_available: Math.floor(Math.random() * 6) + 2,
      avg_battery_health: +(72 + Math.random() * 18).toFixed(1),
      thermal_events: Math.floor(Math.random() * 2),
      v2g_energy_kw: +(Math.random() * 40).toFixed(1),
      carbon_saved_kg: +(Math.random() * 4 + 0.5).toFixed(2)
    };
  }

  // KPI Cards
  setText("kpi-load",      d.current_load_kw.toFixed(0));
  setText("kpi-health",    d.avg_battery_health.toFixed(1));
  setText("kpi-price",     d.price_per_kwh.toFixed(2));
  setText("kpi-sessions",  d.active_sessions);
  setText("kpi-available", d.stations_available);
  setText("kpi-carbon",    d.carbon_saved_kg);
  setText("kpi-v2g",       d.v2g_energy_kw);

  // Load status badge
  const loadPct = d.current_load_kw / 12000 * 100;
  setTextClass("kpi-load-status",
    loadPct > 70 ? "Critical" : loadPct > 45 ? "High" : "Normal",
    loadPct > 70 ? "badge-red" : loadPct > 45 ? "badge-yellow" : "badge-cyan"
  );

  // Price tier badge
  const tier = d.price_per_kwh > 12 ? "Peak" : d.price_per_kwh > 7 ? "Standard" : "Off-Peak";
  setTextClass("kpi-price-tier", tier, tier === "Peak" ? "badge-red" : tier === "Standard" ? "badge-yellow" : "badge-green");

  // Health badge
  const hBadge = d.avg_battery_health > 75 ? "Good" : "Degraded";
  setTextClass("kpi-health-badge", hBadge, hBadge === "Good" ? "badge-green" : "badge-yellow");

  // Gauges
  const lPct = Math.round(loadPct);
  setText("gauge-load-val",    lPct);
  setText("gauge-health-val",  Math.round(d.avg_battery_health));
  setText("gauge-price-val",   d.price_per_kwh.toFixed(1));
  setText("gauge-thermal-val", d.thermal_events > 0 ? "HIGH" : "LOW");
  updateGauge(gaugeLoad,   lPct, 100);
  updateGauge(gaugeHealth, Math.round(d.avg_battery_health), 100);
  updateGauge(gaugePrice,  Math.min(d.price_per_kwh, 20), 20);

  // Line charts rolling update
  loadHistory.shift();  loadHistory.push(d.current_load_kw);
  priceHistory.shift(); priceHistory.push(d.price_per_kwh);
  if (chartLoad)  { chartLoad.data.datasets[0].data  = [...loadHistory];  chartLoad.update("none"); }
  if (chartPrice) { chartPrice.data.datasets[0].data = [...priceHistory]; chartPrice.update("none"); }
}

// ── Carbon Stats ──────────────────────────────────────────────────────────────
async function fetchCarbonStats() {
  let d;
  try {
    const res = await fetch(`${API}/carbon_stats`);
    d = await res.json();
  } catch {
    d = { today_emissions_kg: 28.4, monthly_emissions_kg: 612, v2g_saved_kg: 18.3, equivalent_trees: 3.2 };
  }
  setText("v2g-today",   d.today_emissions_kg);
  setText("v2g-monthly", d.monthly_emissions_kg);
  setText("v2g-saved",   d.v2g_saved_kg);
  setText("v2g-trees",   d.equivalent_trees);
}

// ── Utility helpers ───────────────────────────────────────────────────────────
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}
function setTextClass(id, text, cls) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.className = "kpi-badge " + cls;
}
function showResult(id, html) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = html;
  el.classList.add("visible");
}

// ── PREDICTION CALLS ──────────────────────────────────────────────────────────
async function predictHealth() {
  const payload = {
    battery_level: +v("h-battery"), charging_cycles: +v("h-cycles"),
    temperature: +v("h-temp"), voltage: +v("h-voltage"),
    energy_consumed: +v("h-energy"), thermal_risk_score: +v("h-thermal"),
    time_of_day: new Date().getHours()
  };
  let d;
  try { const r = await fetch(`${API}/predict_health`, postOpts(payload)); d = await r.json(); }
  catch { d = simulateHealth(payload); }
  const col = d.battery_health_status === "Good" ? "var(--accent-green)" : "var(--accent-red)";
  showResult("result-health", `
    <div style="color:${col};font-weight:700;font-size:1.1rem">⚡ ${d.battery_health_status}</div>
    ${d.confidence ? `<div>Confidence: <strong>${d.confidence}%</strong></div>` : ""}
    <div style="color:var(--text-secondary);margin-top:6px">${d.recommendation}</div>
  `);
}

async function predictPriceLoad() {
  const payload = {
    time_of_day: +v("p-hour"), day_of_week: +v("p-dow"), month: new Date().getMonth() + 1,
    grid_demand: +v("p-grid"), current_load: +v("p-load"),
    battery_health: +v("p-health"), station_id_enc: +v("p-station"),
    energy_consumed: 25, charging_time: 40, battery_level: 60
  };
  let price, load;
  try {
    const [pr, lr] = await Promise.all([
      fetch(`${API}/predict_price`, postOpts(payload)).then(r => r.json()),
      fetch(`${API}/predict_load`,  postOpts(payload)).then(r => r.json())
    ]);
    price = pr; load = lr;
  } catch { price = simulatePrice(payload); load = simulateLoad(payload); }
  showResult("result-price", `
    <div>💰 Price: <strong style="color:var(--accent-yellow)">₹${price.price_per_kwh}/kWh</strong>
      <span class="kpi-badge badge-yellow" style="margin-left:8px">${price.pricing_tier}</span></div>
    <div>⚡ Load: <strong style="color:var(--accent-cyan)">${load.predicted_load_kw} kW</strong>
      <span class="kpi-badge badge-cyan" style="margin-left:8px">${load.load_status}</span></div>
    <div style="color:var(--text-secondary);margin-top:6px">Grid stress: ${load.grid_stress_pct}%</div>
    <div style="color:var(--text-secondary)">${price.best_time_to_charge}</div>
  `);
}

async function findNearestStation() {
  const payload = { user_lat: +v("ns-lat"), user_lon: +v("ns-lon"), top_k: +v("ns-topk") };
  let d;
  try { const r = await fetch(`${API}/nearest_station`, postOpts(payload)); d = await r.json(); }
  catch { d = simulateNearest(payload); }
  const rows = d.nearest_stations.map(s =>
    `<div style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.06)">
      📍 <strong>${s.station_id}</strong> — ${s.distance_km} km
      <span class="kpi-badge ${s.availability === "Available" ? "badge-green" : "badge-yellow"}" style="margin-left:6px">${s.availability}</span><br>
      <span style="color:var(--text-secondary);font-size:0.8rem">₹${s.avg_price}/kWh · Load: ${s.avg_load_kw} kW</span>
    </div>`
  ).join("");
  showResult("result-station", rows);
}

async function predictThermal() {
  const payload = {
    temperature: +v("t-temp"), voltage: +v("t-voltage"),
    battery_health: +v("t-health"), charging_cycles: +v("t-cycles"),
    current_load: +v("t-load"), energy_consumed: +v("t-energy"), charging_time: 40
  };
  let d;
  try { const r = await fetch(`${API}/predict_thermal`, postOpts(payload)); d = await r.json(); }
  catch { d = simulateThermal(payload); }
  const col = d.safety_status === "SAFE" ? "var(--accent-green)" : d.safety_status === "WARNING" ? "var(--accent-yellow)" : "var(--accent-red)";
  showResult("result-thermal", `
    <div style="color:${col};font-weight:700;font-size:1.1rem">🛡️ ${d.safety_status}</div>
    <div>Thermal Risk Score: <strong>${d.thermal_risk_score}/100</strong></div>
    <div>Temperature: <strong>${d.temperature_c}°C</strong></div>
    ${d.auto_cutoff_activated ? '<div style="color:var(--accent-red);font-weight:600;margin-top:6px">🚨 AUTO CUT-OFF ACTIVATED!</div>' : ""}
    ${(d.alerts || []).map(a => `<div style="color:var(--accent-yellow);margin-top:4px">${a}</div>`).join("")}
  `);
  // Update safety ring
  updateSafetyRing(d.safety_status, d.alerts || []);
}

function updateSafetyRing(status, alerts) {
  const ring = document.getElementById("safety-ring");
  const list = document.getElementById("alert-list");
  if (!ring) return;
  const map = {
    SAFE:    { cls: "safe",   icon: "✅", label: "SAFE" },
    WARNING: { cls: "warn",   icon: "⚠️", label: "WARNING" },
    DANGER:  { cls: "danger", icon: "🚨", label: "DANGER" }
  };
  const info = map[status] || map.SAFE;
  ring.className = `safety-status-ring ${info.cls}`;
  ring.innerHTML = `<span class="ring-icon">${info.icon}</span><span class="ring-label">${info.label}</span>`;
  if (list) {
    list.innerHTML = alerts.length
      ? alerts.map(a => `<li class="alert-item">${a}</li>`).join("")
      : `<li class="alert-item info">ℹ️ All systems operating within safe parameters.</li>`;
  }
}

// ── Chatbot ───────────────────────────────────────────────────────────────────
async function sendChat() {
  const inp = document.getElementById("chat-input");
  const msg = inp.value.trim();
  if (!msg) return;
  inp.value = "";
  appendBubble(msg, "user");
  let d;
  try { const r = await fetch(`${API}/chatbot`, postOpts({ message: msg })); d = await r.json(); }
  catch { d = { response: localChatbot(msg) }; }
  appendBubble(d.response, "bot");
}

function appendBubble(text, type) {
  const box = document.getElementById("chat-messages");
  if (!box) return;
  const div = document.createElement("div");
  div.className = `chat-bubble ${type}`;
  div.textContent = text;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function localChatbot(msg) {
  const m = msg.toLowerCase();
  if (m.includes("price"))   return "Current price depends on demand. Off-peak (11 PM–5 AM): ₹4–6/kWh. Peak hours can reach ₹12–18/kWh.";
  if (m.includes("battery")) return "Healthy battery: health > 70%. Avoid overcharging and keep temperature below 40°C.";
  if (m.includes("thermal") || m.includes("safe")) return "Our AI monitors temperature & voltage. Auto cut-off triggers at >65°C or risk score >75.";
  if (m.includes("station")) return "Use the map above or the Nearest Station finder to locate the closest charging point.";
  if (m.includes("v2g"))     return "V2G feeds energy back to the grid when your battery > 80% and demand is high. You earn credits!";
  if (m.includes("carbon"))  return "EV charging emits ~0.82 kg CO₂/kWh on India's grid. V2G and off-peak charging reduce this significantly.";
  return "I can help with: charging prices, battery health, thermal safety, stations, V2G, and carbon tracking.";
}

// ── Simulation fallbacks (when Flask offline) ─────────────────────────────────
function simulateHealth(p) {
  const good = p.battery_level > 40 && p.charging_cycles < 800 && p.temperature < 45;
  return { battery_health_status: good ? "Good" : "Poor", confidence: +(75 + Math.random() * 20).toFixed(1),
    recommendation: good ? "Battery is healthy. Continue normal charging." : "Battery health is poor. Consider service." };
}
function simulatePrice(p) {
  const base = p.grid_demand > 7000 ? 13 : 7;
  const price = +(base + (Math.random() - 0.5) * 2).toFixed(2);
  return { price_per_kwh: price, pricing_tier: price > 12 ? "Peak" : price > 7 ? "Standard" : "Off-Peak",
    best_time_to_charge: "11 PM – 5 AM (Off-Peak rates)" };
}
function simulateLoad(p) {
  const load = +(p.grid_demand * 0.55 + Math.random() * 200).toFixed(1);
  return { predicted_load_kw: load, load_status: load > 7000 ? "Critical" : load > 4500 ? "High" : "Normal",
    grid_stress_pct: +(load / 12000 * 100).toFixed(1) };
}
function simulateNearest(p) {
  const stations = Array.from({ length: +p.top_k }, (_, i) => ({
    station_id: `STN_0${i + 1}`, distance_km: +(0.5 + i * 1.2 + Math.random() * 0.5).toFixed(2),
    avg_price: +(6 + Math.random() * 4).toFixed(2), avg_load_kw: +(1000 + Math.random() * 2000).toFixed(0),
    availability: i === 0 ? "Available" : Math.random() > 0.4 ? "Available" : "Busy"
  }));
  return { nearest_stations: stations };
}
function simulateThermal(p) {
  const risk = Math.min(((p.temperature - 25) / 50) * 60 + (p.charging_cycles > 800 ? 15 : 0), 100);
  const status = risk > 75 ? "DANGER" : risk > 50 ? "WARNING" : "SAFE";
  return { thermal_risk_score: +risk.toFixed(1), safety_status: status, temperature_c: p.temperature,
    auto_cutoff_activated: p.temperature > 65 || risk > 75,
    alerts: risk > 50 ? ["⚠️ Elevated thermal risk detected. Monitor closely."] : [] };
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function v(id) { return document.getElementById(id)?.value || 0; }
function postOpts(body) { return { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }; }

// ── Init All ──────────────────────────────────────────────────────────────────
function initAll() {
  // Gauges
  gaugeLoad    = makeGauge("gauge-load",    0, 100, "#00d4ff");
  gaugeHealth  = makeGauge("gauge-health",  0, 100, "#00ff88");
  gaugePrice   = makeGauge("gauge-price",   0, 20,  "#ffd60a");
  gaugeThermal = makeGauge("gauge-thermal", 0, 100, "#ff3b5c");

  // Charts
  const hours24 = Array.from({ length: 12 }, (_, i) => `${(i * 2).toString().padStart(2, "0")}:00`);
  chartLoad  = makeLineChart("chart-load",  "Load (kW)",    "rgb(0,212,255)",   hours24);
  chartPrice = makeLineChart("chart-price", "Price (₹/kWh)","rgb(255,214,10)",  hours24);
  makeLSTMChart();
  makeHealthDistChart();

  // Map
  initMap();

  // Data
  fetchStations();
  fetchLiveData();
  fetchCarbonStats();

  // Polling
  setInterval(fetchLiveData,   4000);
  setInterval(fetchCarbonStats, 30000);
  setInterval(fetchStations,   60000);
}
