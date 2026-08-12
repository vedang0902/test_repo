/**
 * PaymentPipeline Dashboard — Core App
 * Handles routing, API client, and shared state.
 */

const API_BASE = window.location.origin;

// ── API Client ─────────────────────────────────────────────────────────────

const api = {
  async get(path) {
    try {
      const res = await fetch(`${API_BASE}${path}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      console.error(`API GET ${path} failed:`, e.message);
      return null;
    }
  },

  async post(path, body) {
    try {
      const res = await fetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      return await res.json();
    } catch (e) {
      console.error(`API POST ${path} failed:`, e.message);
      return null;
    }
  },
};

// ── Router ─────────────────────────────────────────────────────────────────

const pages = document.querySelectorAll('.page');
const navItems = document.querySelectorAll('.nav-item');

function navigateTo(pageId) {
  pages.forEach(p => p.classList.remove('active'));
  navItems.forEach(n => n.classList.remove('active'));

  const page = document.getElementById(`page-${pageId}`);
  const navItem = document.querySelector(`[data-page="${pageId}"]`);

  if (page) page.classList.add('active');
  if (navItem) navItem.classList.add('active');

  document.getElementById('page-title').textContent =
    navItem ? navItem.textContent.trim() : pageId;

  // Trigger page-specific data load
  if (pageId === 'transactions') loadTransactions();
  if (pageId === 'fraud') loadFraudStats();
  if (pageId === 'webhooks') loadWebhookStats();
  if (pageId === 'reconciliation') loadReconciliation();
  if (pageId === 'accounts') loadAccounts();
}

navItems.forEach(item => {
  item.addEventListener('click', () => navigateTo(item.dataset.page));
});

// ── Error feed ─────────────────────────────────────────────────────────────

const errorFeed = document.getElementById('error-feed');
let errorCount = 0;

window.addErrorFeedItem = function(message, level = 'error') {
  const placeholder = errorFeed.querySelector('.feed-placeholder');
  if (placeholder) placeholder.remove();

  errorCount++;
  document.getElementById('error-count-badge').textContent = errorCount;

  const item = document.createElement('div');
  item.className = `feed-item ${level}`;
  const ts = new Date().toLocaleTimeString();
  item.textContent = `[${ts}] ${message}`;

  errorFeed.insertBefore(item, errorFeed.firstChild);

  // Keep feed bounded
  while (errorFeed.children.length > 50) {
    errorFeed.removeChild(errorFeed.lastChild);
  }
};

// ── Health poll ────────────────────────────────────────────────────────────

async function pollHealth() {
  const health = await api.get('/health');
  if (!health) return;

  const chip = document.getElementById('health-chip');
  const label = document.getElementById('health-label');
  const dot = chip.querySelector('.dot');
  const banner = document.getElementById('alert-banner');
  const alertText = document.getElementById('alert-text');

  label.textContent = health.status.toUpperCase();
  dot.className = 'dot';

  if (health.status === 'ok') {
    dot.classList.add('dot-green');
  } else if (health.status === 'degraded') {
    dot.classList.add('dot-orange');
    banner.style.display = 'flex';
    alertText.textContent = 'Pipeline degraded — webhook storms or reconciliation mismatch detected';
  } else if (health.status === 'critical') {
    dot.classList.add('dot-red');
    banner.style.display = 'flex';
    alertText.textContent = 'CRITICAL: Pipeline in error state — DB unavailable or severe reconciliation drift';
  }
}

// ── Refresh button ─────────────────────────────────────────────────────────

document.getElementById('btn-refresh').addEventListener('click', () => {
  dashboardRefresh();
  pollHealth();
});

// ── Init ───────────────────────────────────────────────────────────────────

navigateTo('dashboard');
pollHealth();
setInterval(pollHealth, 10000);
