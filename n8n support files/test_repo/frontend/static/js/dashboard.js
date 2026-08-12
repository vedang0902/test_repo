/**
 * Dashboard page — KPI cards, DB pool visual, bug signal counters.
 * Polls /analytics/summary and /health every 5 seconds.
 */

function fmtCurrency(v) {
  if (v === null || v === undefined) return '—';
  return '$' + Number(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtNum(v) {
  if (v === null || v === undefined) return '—';
  return Number(v).toLocaleString();
}

async function dashboardRefresh() {
  const [summary, health] = await Promise.all([
    api.get('/analytics/summary'),
    api.get('/health'),
  ]);

  if (summary) {
    document.getElementById('kpi-total').textContent     = fmtNum(summary.total_transactions);
    document.getElementById('kpi-completed').textContent = fmtNum(summary.successful_transactions);
    document.getElementById('kpi-failed').textContent    = fmtNum(summary.failed_transactions + summary.partial_commits);
    document.getElementById('kpi-fraud').textContent     = fmtNum(summary.fraud_blocked);
    document.getElementById('kpi-volume').textContent    = fmtCurrency(summary.total_volume_usd);
    document.getElementById('kpi-drift').textContent     = `$${summary.reconciliation_drift.toFixed(6)}`;

    const pct = summary.total_transactions > 0
      ? ((summary.successful_transactions / summary.total_transactions) * 100).toFixed(1)
      : '0';
    document.getElementById('kpi-completed-pct').textContent = `${pct}% success rate`;

    // Bug signal counters (from summary)
    document.getElementById('bug-partial-count').textContent  = fmtNum(summary.partial_commits);
    document.getElementById('bug-storm-count').textContent    = fmtNum(summary.webhook_retry_storms);
    document.getElementById('bug-deadlock-count').textContent = fmtNum(summary.deadlock_events);

    // Highlight if non-zero
    if (summary.partial_commits > 0)    addErrorFeedItem(`Partial commit: ${summary.partial_commits} orphaned debits`, 'critical');
    if (summary.webhook_retry_storms > 0) addErrorFeedItem(`Webhook retry storm: ${summary.webhook_retry_storms} candidates`, 'warn');
  }

  if (health) {
    // DB pool
    const avail = health.db_pool_available;
    const inuse = health.db_pool_in_use;
    const total = avail + inuse;

    document.getElementById('pool-avail').textContent    = avail;
    document.getElementById('pool-inuse').textContent    = inuse;
    document.getElementById('pool-max').textContent      = total;
    document.getElementById('pool-exhausted').textContent = '—';

    const badge = document.getElementById('pool-status-badge');
    if (avail === 0) {
      badge.textContent = 'EXHAUSTED';
      badge.className = 'badge badge-red';
      addErrorFeedItem('DB connection pool exhausted — all connections in use', 'critical');
    } else if (avail < 2) {
      badge.textContent = 'LOW';
      badge.className = 'badge badge-yellow';
    } else {
      badge.textContent = 'OK';
      badge.className = 'badge';
    }

    // Pool visual
    const poolVisual = document.getElementById('pool-visual');
    poolVisual.innerHTML = '';
    for (let i = 0; i < total; i++) {
      const conn = document.createElement('div');
      conn.className = `pool-conn ${i < inuse ? 'in-use' : 'available'}`;
      conn.textContent = i < inuse ? 'IN' : 'OK';
      poolVisual.appendChild(conn);
    }

    // Component health to error feed
    const components = health.components || {};
    Object.entries(components).forEach(([comp, status]) => {
      if (status !== 'ok') {
        addErrorFeedItem(`${comp}: ${status}`, status === 'critical' ? 'critical' : 'warn');
      }
    });
  }

  // Fraud + webhook + generator stats
  const [fraudStats, wbStats, genStats] = await Promise.all([
    api.get('/analytics/fraud'),
    api.get('/analytics/webhooks'),
    api.get('/analytics/generator'),
  ]);

  if (fraudStats) {
    document.getElementById('bug-cascade-count').textContent = fmtNum(fraudStats.flagged_account_count);
    document.getElementById('bug-idem-count').textContent    = '—'; // From Prometheus
  }

  if (wbStats) {
    document.getElementById('bug-storm-count').textContent = fmtNum(wbStats.storm_candidates);
  }

  if (genStats) {
    document.getElementById('bug-cache-count').textContent = fmtNum(genStats.cache_size);
    if (genStats.cache_size > 500) {
      addErrorFeedItem(`Memory leak: transaction cache at ${genStats.cache_size} entries`, 'warn');
    }
  }
}

// Auto-refresh every 5 seconds
dashboardRefresh();
setInterval(dashboardRefresh, 5000);
