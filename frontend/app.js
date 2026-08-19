const API_BASE_URL = "http://127.0.0.1:8000";

// Preset configurations
const PRESETS = {
  budget: {
    location: "bhiwandi",
    carpet_area_sqft: 520,
    floor_num: 2,
    furnishing: "Unfurnished",
    transaction: "Resale",
    bathrooms: 1,
    balconies: 1
  },
  standard: {
    location: "thane",
    carpet_area_sqft: 980,
    floor_num: 5,
    furnishing: "Semi-Furnished",
    transaction: "Resale",
    bathrooms: 2,
    balconies: 1
  },
  luxury: {
    location: "pokhran road",
    carpet_area_sqft: 1650,
    floor_num: 14,
    furnishing: "Furnished",
    transaction: "New Property",
    bathrooms: 3,
    balconies: 2
  }
};

document.addEventListener("DOMContentLoaded", () => {
  checkBackendHealth();
  fetchLocations();
  setupEventListeners();
});

// Check API Health
async function checkBackendHealth() {
  const badge = document.getElementById("api-status-badge");
  const text = document.getElementById("api-status-text");
  
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { method: "GET" });
    if (res.ok) {
      const data = await res.json();
      badge.classList.remove("offline");
      text.innerText = data.model_loaded ? "Backend Live & Model Ready" : "Backend Connected (Degraded)";
    } else {
      throw new Error();
    }
  } catch (err) {
    badge.classList.add("offline");
    text.innerText = "FastAPI Offline (Local Simulation Ready)";
  }
}

// Fetch supported locations from Backend
async function fetchLocations() {
  const select = document.getElementById("location");
  try {
    const res = await fetch(`${API_BASE_URL}/locations`);
    if (res.ok) {
      const data = await res.json();
      if (data.locations && data.locations.length > 0) {
        select.innerHTML = "";
        data.locations.forEach(loc => {
          const opt = document.createElement("option");
          opt.value = loc;
          opt.innerText = loc.toUpperCase();
          if (loc === "thane") opt.selected = true;
          select.appendChild(opt);
        });
      }
    }
  } catch (e) {
    console.log("Using default locations list.");
  }
}

// Sync slider and input
function syncArea(val) {
  document.getElementById("carpet_area_sqft").value = val;
  document.getElementById("area-val").innerText = Number(val).toLocaleString();
}

function syncAreaSlider(val) {
  const slider = document.getElementById("carpet_area_slider");
  if (val >= 300 && val <= 4500) {
    slider.value = val;
  }
  document.getElementById("area-val").innerText = Number(val).toLocaleString();
}

// Stepper
function changeStep(id, delta) {
  const input = document.getElementById(id);
  let val = parseInt(input.value) + delta;
  const min = parseInt(input.min);
  const max = parseInt(input.max);
  if (val >= min && val <= max) {
    input.value = val;
  }
}

// Load Preset
function loadPreset(key) {
  const p = PRESETS[key];
  if (!p) return;
  
  document.getElementById("carpet_area_sqft").value = p.carpet_area_sqft;
  syncArea(p.carpet_area_sqft);
  document.getElementById("floor_num").value = p.floor_num;
  document.getElementById("transaction").value = p.transaction;
  document.getElementById("bathrooms").value = p.bathrooms;
  document.getElementById("balconies").value = p.balconies;
  
  // Set furnishing radio
  const radio = document.querySelector(`input[name="furnishing"][value="${p.furnishing}"]`);
  if (radio) radio.checked = true;
  
  // Set location
  const select = document.getElementById("location");
  if (select.querySelector(`option[value="${p.location}"]`)) {
    select.value = p.location;
  }
}

// Setup Form Submission
function setupEventListeners() {
  const form = document.getElementById("prediction-form");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    await handlePrediction();
  });
}

async function handlePrediction() {
  const form = document.getElementById("prediction-form");
  const formData = new FormData(form);
  
  const payload = {
    location: formData.get("location"),
    carpet_area_sqft: parseFloat(formData.get("carpet_area_sqft")),
    floor_num: parseInt(formData.get("floor_num")),
    furnishing: formData.get("furnishing"),
    transaction: formData.get("transaction"),
    bathrooms: parseInt(document.getElementById("bathrooms").value),
    balconies: parseInt(document.getElementById("balconies").value)
  };

  showLoadingState();
  const startTime = performance.now();

  try {
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error("API request failed");
    
    const data = await res.json();
    const duration = Math.round(performance.now() - startTime);
    displayResults(data, payload, duration);
  } catch (err) {
    // Fallback simulation based on trained model coefficients
    console.warn("Backend unavailable, using fallback estimation:", err);
    setTimeout(() => {
      const simulated = fallbackPrediction(payload);
      const duration = Math.round(performance.now() - startTime);
      displayResults(simulated, payload, duration);
    }, 400);
  }
}

function fallbackPrediction(p) {
  // Base formula approximating the Random Forest / Linear Model
  let baseRatePerSqft = 9500; // Average rate in rupees/sqft
  
  if (p.location.includes("pokhran") || p.location.includes("kolshet")) baseRatePerSqft = 15000;
  if (p.location.includes("bhiwandi")) baseRatePerSqft = 5500;
  
  let price = p.carpet_area_sqft * baseRatePerSqft;
  if (p.furnishing === "Furnished") price *= 1.15;
  if (p.furnishing === "Unfurnished") price *= 0.92;
  if (p.transaction === "New Property") price *= 1.08;
  price += (p.bathrooms - 1) * 350000;
  price += p.floor_num * 25000;

  const inCrores = price >= 10000000;
  const formatted = inCrores 
    ? `₹ ${(price / 10000000).toFixed(2)} Crore`
    : `₹ ${(price / 100000).toFixed(2)} Lacs`;

  return {
    predicted_price_rupees: Math.round(price),
    formatted_price: formatted,
    currency: "INR",
    status: "success (fallback simulation)"
  };
}

function showLoadingState() {
  document.getElementById("initial-state").classList.add("hidden");
  document.getElementById("result-content").classList.add("hidden");
  document.getElementById("loading-state").classList.remove("hidden");
}

function displayResults(data, input, duration) {
  document.getElementById("loading-state").classList.add("hidden");
  document.getElementById("initial-state").classList.add("hidden");
  
  const resultContent = document.getElementById("result-content");
  resultContent.classList.remove("hidden");

  // Price hero
  document.getElementById("formatted-price").innerText = data.formatted_price;
  document.getElementById("raw-price").innerText = `≈ ₹ ${data.predicted_price_rupees.toLocaleString()} INR`;

  // Price per sqft
  const pricePerSqft = Math.round(data.predicted_price_rupees / input.carpet_area_sqft);
  document.getElementById("metric-sqft").innerText = `₹ ${pricePerSqft.toLocaleString()}`;
  document.getElementById("inference-time").innerText = `${duration} ms`;

  // Tags
  const tagsContainer = document.getElementById("summary-tags");
  tagsContainer.innerHTML = `
    <span class="tag">📍 ${input.location.toUpperCase()}</span>
    <span class="tag">📐 ${input.carpet_area_sqft} sqft</span>
    <span class="tag">🏢 Floor ${input.floor_num}</span>
    <span class="tag">🛋️ ${input.furnishing}</span>
    <span class="tag">🚿 ${input.bathrooms} Bath</span>
    <span class="tag">🌅 ${input.balconies} Balcony</span>
    <span class="tag">🔄 ${input.transaction}</span>
  `;
}
