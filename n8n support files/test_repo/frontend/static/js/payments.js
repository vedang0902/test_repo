/**
 * Transactions, accounts, fraud, webhooks, reconciliation page loaders.
 */

// ── Transactions ───────────────────────────────────────────────────────────

async function loadTransactions() {
  const status = document.getElementById('tx-filter-status').value;
  const url = status ? `/payments?status=${status}&limit=100` : '/payments?limit=100';
  const data = await api.get(url);
  const tbody = document.getElementById('tx-tbody');

  if (!data || data.length === 0) {
    tbody.innerHTML = '<tr><td colspan="10" class="table-empty">No transactions yet</td></tr>';
    return;
  }

  tbody.innerHTML = data.map(tx => `
    <tr>
      <td title="${tx.id}">${tx.id.slice(0, 8)}…</td>
      <td>${tx.from_account}</td>
      <td>${tx.to_account}</td>
      <td>$${Number(tx.amount).toFixed(2)}</td>
      <td>${tx.currency}</td>
      <td>${tx.method}</td>
      <td><span class="status-pill status-${tx.status}">${tx.status.replace('_', ' ')}</span></td>
      <td style="color: ${tx.fraud_score > 0.5 ? '#f44f4f' : '#8892b0'}">${tx.fraud_score.toFixed(3)}</td>
      <td>$${Number(tx.fee).toFixed(4)}</td>
      <td>${new Date(tx.created_at).toLocaleTimeString()}</td>
    </tr>
  `).join('');
}

document.getElementById('tx-filter-status')?.addEventListener('change', loadTransactions);

// ── Accounts ───────────────────────────────────────────────────────────────

async function loadAccounts() {
  const data = await api.get('/accounts');
  const tbody = document.getElementById('accounts-tbody');

  if (!data || data.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="table-empty">No accounts found</td></tr>';
    return;
  }

  tbody.innerHTML = data.map(a => `
    <tr>
      <td>${a.id}</td>
      <td>${a.name}</td>
      <td>$${Number(a.balance).toLocaleString('en-US', { minimumFractionDigits: 2 })}</td>
      <td>${a.currency}</td>
      <td><span class="status-pill ${a.is_active ? 'status-completed' : 'status-failed'}">${a.is_active ? 'Active' : 'Inactive'}</span></td>
    </tr>
  `).join('');
}

// ── Fraud Stats ────────────────────────────────────────────────────────────

async function loadFraudStats() {
  const data = await api.get('/analytics/fraud');
  if (!data) return;

  const accountsList = document.getElementById('fraud-accounts-list');
  if (data.high_risk_accounts && data.high_risk_accounts.length > 0) {
    accountsList.innerHTML = data.high_risk_accounts.map(a => `
      <div class="list-item">
        <span class="item-id">${a.account_id}</span>
        <span class="item-val" style="color: ${a.score > 0.65 ? '#f44f4f' : '#f97316'}">
          score: ${Number(a.score).toFixed(3)}
        </span>
      </div>
    `).join('');
  } else {
    accountsList.innerHTML = '<div class="feed-placeholder">No high-risk accounts yet</div>';
  }

  const distPanel = document.getElementById('fraud-score-dist');
  distPanel.innerHTML = `
    <div class="config-row"><span>Accounts tracked</span><span class="mono">${data.score_accumulator_entries}</span></div>
    <div class="config-row"><span>Flagged accounts</span><span class="mono" style="color:#f44f4f">${data.flagged_account_count}</span></div>
    <div class="config-row"><span>Total checks</span><span class="mono">${data.check_count_entries}</span></div>
  `;
}

// ── Webhook Stats ──────────────────────────────────────────────────────────

async function loadWebhookStats() {
  const data = await api.get('/analytics/webhooks');
  if (!data) return;

  document.getElementById('wh-total').textContent  = data.total_webhooks_seen || '—';
  document.getElementById('wh-dupes').textContent  = data.total_processed_in_memory || '—';
  document.getElementById('wh-storms').textContent = data.storm_candidates || '—';

  const list = document.getElementById('storm-candidates-list');
  if (data.storm_candidates && data.storm_candidates.length > 0) {
    list.innerHTML = data.storm_candidates.map(s => `
      <div class="list-item">
        <span class="item-id">${s.webhook_id}</span>
        <span class="item-val">processed ${s.count}×</span>
      </div>
    `).join('');
  } else {
    list.innerHTML = '<div class="feed-placeholder">No storm candidates detected yet</div>';
  }
}

// ── Reconciliation ─────────────────────────────────────────────────────────

async function loadReconciliation() {
  const data = await api.get('/analytics/reconciliation');
  if (!data) return;

  document.getElementById('recon-ledger').textContent    = `$${Number(data.ledger_balance).toFixed(6)}`;
  document.getElementById('recon-actual').textContent    = `$${Number(data.actual_balance).toFixed(6)}`;
  document.getElementById('recon-drift').textContent     = `$${Number(data.drift).toFixed(8)}`;
  document.getElementById('recon-drift-pct').textContent = `${Number(data.drift_pct).toFixed(6)}%`;
  document.getElementById('recon-count').textContent     = data.transaction_count;

  const statusEl = document.getElementById('recon-status');
  statusEl.textContent = data.status.toUpperCase();
  statusEl.className = `mono ${data.status === 'ok' ? '' : data.status === 'critical' ? 'badge-red-text' : ''}`;
}
