<script>
  import { onMount } from 'svelte';
  import { KeyRound, Layers, RefreshCw, Plus, Shield, AlertTriangle, Server, Pencil, Trash2, X, Search, Radar, ChevronDown, ChevronUp, CheckCircle, XCircle, Clock } from '@lucide/svelte';
  import { appState } from '$lib/state.svelte.js';
  import Button from '$lib/components/Button.svelte';
  import Input from '$lib/components/Input.svelte';
  import Card from '$lib/components/Card.svelte';
  import Modal from '$lib/components/Modal.svelte';

  // ─── Local State ──────────────────────────────────────────────────────────
  let providerCredentials = $state([]);
  let totalCount = $state(0);
  let providerPools = $state([]);
  let providerLoading = $state(false);
  let loadingMore = $state(false);
  let providerError = $state('');

  // Pagination / lazy-loading
  const PAGE_SIZE = 100;
  let currentPage = $state(0);
  let hasMore = $state(false);

  // Filtering
  let searchQuery = $state('');
  let providerFilter = $state('');
  let searchTimer = null;

  // Virtualization
  const ROW_HEIGHT = 45;
  const OVERSCAN = 8;
  let scrollTop = $state(0);
  let viewportHeight = $state(0);
  let vscrollEl;

  let visibleRange = $derived.by(() => {
    const loaded = providerCredentials.length;
    if (loaded === 0 || viewportHeight === 0) {
      return { start: 0, end: 0, padTop: 0, padBottom: 0 };
    }
    const start = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN);
    const visibleCount = Math.ceil(viewportHeight / ROW_HEIGHT) + OVERSCAN * 2;
    const end = Math.min(loaded, start + visibleCount);
    return {
      start,
      end,
      padTop: start * ROW_HEIGHT,
      padBottom: (loaded - end) * ROW_HEIGHT
    };
  });

  let visibleItems = $derived(providerCredentials.slice(visibleRange.start, visibleRange.end));

  // Add/Edit modals
  let showAddProviderModal = $state(false);
  let addProviderTab = $state('autodiscovery'); // 'standard' | 'autodiscovery'
  let addProviderForm = $state({ pool_id: '', provider: 'openai', api_key: '', base_url: 'https://api.openai.com', weight: 1 });
  let addProviderLoading = $state(false);

  // Auto-discovery form
  let autoDiscoverForm = $state({ provider: 'custom', api_key: '', base_url: '', weight: 1, label: '', account_id: '', api_token: '' });
  let customKeyMode = $state('single'); // 'single' | 'batch'
  let batchKeysText = $state('');
  let batchReport = $state(null);
  let autoDiscoverLoading = $state(false);

  let parsedBatchKeys = $derived.by(() => {
    if (!batchKeysText.trim()) return [];
    const lines = batchKeysText.split(/[\r\n,]+/);
    const seen = new Set();
    const result = [];
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed && !seen.has(trimmed)) {
        seen.add(trimmed);
        result.push(trimmed);
      }
    }
    return result;
  });

  let parsedBatchKeysCount = $derived(parsedBatchKeys.length);

  // Edit modal
  let showEditModal = $state(false);
  let editForm = $state({ id: 0, provider: '', api_key: '', base_url: '', weight: 1, is_healthy: true });
  let editLoading = $state(false);

  // Delete confirmation
  let showDeleteConfirm = $state(false);
  let deleteTargetId = $state(null);
  let deleteLoading = $state(false);

  // Bulk selection and deletion
  let selectedIds = $state([]);
  let showBulkDeleteConfirm = $state(false);
  let bulkDeleteLoading = $state(false);

  // Refresh all providers
  let refreshLoading = $state(false);

  // Re-discovery state
  let rediscoveryStatus = $state({ status: 'IDLE' });
  let isRediscovering = $state(false);
  let showRediscoveryDetails = $state(false);
  let parsedRediscoveryReport = $state(null);
  let rediscoveryPollInterval = null;

  // ─── Load state on adminKey change ─────────────────────────────────────────
  $effect(() => {
    if (appState.adminKey.trim()) {
      reloadCredentials();
      loadPools();
    }
  });

  // Debounced search/filter → reload first page
  function onSearchInput() {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => reloadCredentials(), 300);
  }

  // ─── API Helper Headers ───────────────────────────────────────────────────
  function adminHeaders() {
    return {
      'Authorization': `Bearer ${appState.getAdminKey()}`,
      'Content-Type': 'application/json'
    };
  }

  async function loadCredentialsPage(page) {
    try {
      const params = new URLSearchParams();
      params.append('limit', String(PAGE_SIZE));
      params.append('offset', String(page * PAGE_SIZE));
      if (searchQuery.trim()) params.append('search', searchQuery.trim());
      if (providerFilter) params.append('provider', providerFilter);

      const res = await fetch(`/api/v1/admin/credentials?${params.toString()}`, { headers: adminHeaders() });
      if (res.ok) {
        const data = await res.json();
        const rows = data.data ?? data;
        return { rows, total: data.total ?? rows.length };
      }
      const err = await res.json();
      return { error: err.error || `Error ${res.status}` };
    } catch (e) {
      return { error: `Network error: ${e.message}` };
    }
  }

  async function reloadCredentials() {
    providerLoading = true;
    providerError = '';
    currentPage = 0;
    appState.apiLoading = true;
    try {
      const result = await loadCredentialsPage(0);
      if (result.error) {
        providerError = result.error;
      } else {
        providerCredentials = result.rows;
        totalCount = result.total;
        hasMore = result.rows.length < result.total;
        selectedIds = selectedIds.filter(id => providerCredentials.some(c => c.id === id));
        if (vscrollEl) vscrollEl.scrollTop = 0;
      }
    } finally {
      providerLoading = false;
      appState.apiLoading = false;
    }
  }

  async function loadMore() {
    if (loadingMore || !hasMore) return;
    loadingMore = true;
    appState.apiLoading = true;
    const nextPage = currentPage + 1;
    try {
      const result = await loadCredentialsPage(nextPage);
      if (result.error) {
        appState.addToast('error', result.error);
      } else {
        providerCredentials = [...providerCredentials, ...result.rows];
        currentPage = nextPage;
        hasMore = providerCredentials.length < result.total;
      }
    } finally {
      loadingMore = false;
      appState.apiLoading = false;
    }
  }

  function onVScroll(e) {
    scrollTop = e.target.scrollTop;
    // Trigger lazy load when near the bottom of the loaded list
    const remaining = providerCredentials.length * ROW_HEIGHT - (scrollTop + viewportHeight);
    if (hasMore && !loadingMore && remaining < ROW_HEIGHT * 10) {
      loadMore();
    }
  }

  async function loadPools() {
    appState.apiLoading = true;
    try {
      const res = await fetch('/api/v1/admin/pools', { headers: adminHeaders() });
      if (res.ok) {
        providerPools = await res.json();
      }
    } catch (e) {
      console.error('Failed to load pools', e);
    } finally {
      appState.apiLoading = false;
    }
  }

  function openAddProviderModal() {
    addProviderForm = { pool_id: '', provider: 'openai', api_key: '', base_url: 'https://api.openai.com', weight: 1 };
    autoDiscoverForm = { provider: 'custom', api_key: '', base_url: '', weight: 1, label: '', account_id: '', api_token: '' };
    customKeyMode = 'single';
    batchKeysText = '';
    batchReport = null;
    addProviderTab = 'autodiscovery';
    showAddProviderModal = true;
    loadPools();
  }

  async function createCredential() {
    addProviderLoading = true;
    appState.apiLoading = true;
    try {
      const res = await fetch('/api/v1/admin/credentials', {
        method: 'POST',
        headers: adminHeaders(),
        body: JSON.stringify({
          pool_id: parseInt(addProviderForm.pool_id),
          provider: addProviderForm.provider,
          api_key: addProviderForm.api_key,
          base_url: addProviderForm.base_url,
          weight: addProviderForm.weight || 1
        })
      });
      if (res.ok || res.status === 201) {
        appState.addToast('success', 'Credential created successfully');
        showAddProviderModal = false;
        reloadCredentials();
      } else {
        const err = await res.json();
        appState.addToast('error', err.details || err.error || 'Failed to create credential');
      }
    } catch (e) {
      appState.addToast('error', `Network error: ${e.message}`);
    } finally {
      addProviderLoading = false;
      appState.apiLoading = false;
    }
  }

  async function autoDiscoverProvider() {
    if ((autoDiscoverForm.provider === 'custom' || autoDiscoverForm.provider === 'puter' || autoDiscoverForm.provider === '1minai') && customKeyMode === 'batch') {
      if (parsedBatchKeys.length === 0) {
        appState.addToast('error', 'Please enter at least one API key in the batch input box');
        return;
      }
    }

    autoDiscoverLoading = true;
    appState.apiLoading = true;
    batchReport = null;
    let endpoint;
    if (autoDiscoverForm.provider === 'nvidia') {
      endpoint = '/api/v1/admin/providers/nvidia';
    } else if (autoDiscoverForm.provider === 'ollama') {
      endpoint = '/api/v1/admin/providers/ollama';
    } else if (autoDiscoverForm.provider === 'openrouter') {
      endpoint = '/api/v1/admin/providers/openrouter';
    } else if (autoDiscoverForm.provider === '1minai') {
      endpoint = '/api/v1/admin/providers/1minai';
    } else if (autoDiscoverForm.provider === 'cloudflare') {
      endpoint = '/api/v1/admin/providers/cloudflare';
    } else if (autoDiscoverForm.provider === 'sarvam') {
      endpoint = '/api/v1/admin/providers/sarvam';
    } else if (autoDiscoverForm.provider === 'puter') {
      endpoint = '/api/v1/admin/providers/puter';
    } else if (autoDiscoverForm.provider === 'agentrouter') {
      endpoint = '/api/v1/admin/providers/agentrouter';
    } else if (autoDiscoverForm.provider === 'zenmux') {
      endpoint = '/api/v1/admin/providers/zenmux';
    } else if (autoDiscoverForm.provider === 'gemini') {
      endpoint = '/api/v1/admin/providers/gemini';
    } else {
      endpoint = '/api/v1/admin/providers/custom';
    }
    try {
      let payload;
      if (autoDiscoverForm.provider === 'cloudflare') {
        // Cloudflare uses a dedicated DTO with account_id + api_token
        payload = {
          account_id: autoDiscoverForm.account_id,
          api_token: autoDiscoverForm.api_token,
          weight: autoDiscoverForm.weight || 1
        };
      } else if (autoDiscoverForm.provider === 'puter' && customKeyMode === 'batch') {
        payload = {
          api_keys: parsedBatchKeys,
          weight: autoDiscoverForm.weight || 1
        };
      } else if (autoDiscoverForm.provider === '1minai' && customKeyMode === 'batch') {
        // 1min.ai bulk registration: backend validates each key one by one and
        // binds every valid key to the full manifest model catalog.
        payload = {
          api_keys: parsedBatchKeys,
          weight: autoDiscoverForm.weight || 1
        };
      } else if (autoDiscoverForm.provider === 'custom' && customKeyMode === 'batch') {
        payload = {
          provider: 'custom',
          api_keys: parsedBatchKeys,
          base_url: autoDiscoverForm.base_url,
          weight: autoDiscoverForm.weight || 1
        };
        if (autoDiscoverForm.label) {
          payload.label = autoDiscoverForm.label;
        }
      } else {
        payload = {
          provider: autoDiscoverForm.provider,
          api_key: autoDiscoverForm.api_key,
          base_url: autoDiscoverForm.base_url,
          weight: autoDiscoverForm.weight || 1
        };
        if (autoDiscoverForm.provider === 'custom' && autoDiscoverForm.label) {
          payload.label = autoDiscoverForm.label;
        }
      }
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: adminHeaders(),
        body: JSON.stringify(payload)
      });
      if (res.ok) {
        const data = await res.json();
        const displayName = autoDiscoverForm.provider === 'custom'
          ? (autoDiscoverForm.label || 'Custom')
          : autoDiscoverForm.provider === '1minai' ? '1min.ai'
          : autoDiscoverForm.provider === 'cloudflare' ? 'Cloudflare Workers AI'
          : autoDiscoverForm.provider === 'sarvam' ? 'Sarvam AI'
          : autoDiscoverForm.provider === 'puter' ? 'Puter.com'
          : autoDiscoverForm.provider === 'agentrouter' ? 'AgentRouter'
          : autoDiscoverForm.provider === 'zenmux' ? 'ZenMux'
          : autoDiscoverForm.provider === 'gemini' ? 'Google AI Studio (Gemini)'
          : autoDiscoverForm.provider.toUpperCase();

        if (data.results && data.results.length > 0) {
          if (data.failed_count > 0) {
            batchReport = data;
            appState.addToast(data.success_count > 0 ? 'warning' : 'error', data.message || `Batch finished: ${data.success_count} succeeded, ${data.failed_count} failed`);
            reloadCredentials();
            appState.loadModels(true);
            return;
          }
        }

        appState.addToast('success', data.message || `Successfully synchronized ${data.models_count || 0} ${displayName} models`);
        showAddProviderModal = false;
        reloadCredentials();
        appState.loadModels(true);
      } else {
        const err = await res.json();
        appState.addToast('error', err.details || err.error || 'Auto-discovery failed');
      }
    } catch (e) {
      appState.addToast('error', `Network error: ${e.message}`);
    } finally {
      autoDiscoverLoading = false;
      appState.apiLoading = false;
    }
  }

  function openEditModal(cred) {
    editForm = {
      id: cred.id,
      provider: cred.provider,
      api_key: '',
      base_url: cred.base_url,
      weight: cred.weight,
      is_healthy: cred.is_healthy
    };
    showEditModal = true;
  }

  async function updateCredential() {
    editLoading = true;
    appState.apiLoading = true;
    try {
      const res = await fetch(`/api/v1/admin/credentials/${editForm.id}`, {
        method: 'PUT',
        headers: adminHeaders(),
        body: JSON.stringify({
          provider: editForm.provider,
          api_key: editForm.api_key || undefined,
          base_url: editForm.base_url,
          weight: editForm.weight,
          is_healthy: editForm.is_healthy
        })
      });
      if (res.ok) {
        appState.addToast('success', 'Credential updated successfully');
        showEditModal = false;
        reloadCredentials();
      } else {
        const err = await res.json();
        appState.addToast('error', err.details || err.error || 'Failed to update credential');
      }
    } catch (e) {
      appState.addToast('error', `Network error: ${e.message}`);
    } finally {
      editLoading = false;
      appState.apiLoading = false;
    }
  }

  function confirmDelete(id) {
    deleteTargetId = id;
    showDeleteConfirm = true;
  }

  async function deleteCredentialById() {
    deleteLoading = true;
    appState.apiLoading = true;
    try {
      const res = await fetch(`/api/v1/admin/credentials/${deleteTargetId}`, {
        method: 'DELETE',
        headers: adminHeaders()
      });
      if (res.ok) {
        appState.addToast('success', 'Credential deleted successfully');
        showDeleteConfirm = false;
        deleteTargetId = null;
        reloadCredentials();
      } else {
        const err = await res.json();
        appState.addToast('error', err.details || err.error || 'Failed to delete credential');
      }
    } catch (e) {
      appState.addToast('error', `Network error: ${e.message}`);
    } finally {
      deleteLoading = false;
      appState.apiLoading = false;
    }
  }

  function confirmBulkDelete() {
    showBulkDeleteConfirm = true;
  }

  async function deleteCredentialsBulk() {
    bulkDeleteLoading = true;
    appState.apiLoading = true;
    try {
      const res = await fetch('/api/v1/admin/credentials/bulk-delete', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...adminHeaders()
        },
        body: JSON.stringify({ ids: selectedIds })
      });
      if (res.ok) {
        appState.addToast('success', `${selectedIds.length} credentials deleted successfully`);
        showBulkDeleteConfirm = false;
        selectedIds = [];
        reloadCredentials();
        loadPools();
        appState.loadModels(true);
      } else {
        const err = await res.json();
        appState.addToast('error', err.details || err.error || 'Failed to delete credentials');
      }
    } catch (e) {
      appState.addToast('error', `Network error: ${e.message}`);
    } finally {
      bulkDeleteLoading = false;
      appState.apiLoading = false;
    }
  }

  function providerBadgeClass(provider) {
    switch ((provider || '').toLowerCase()) {
      case 'openai': return 'badge-openai';
      case 'nvidia': return 'badge-nvidia';
      case 'ollama': return 'badge-ollama';
      case 'anthropic': return 'badge-anthropic';
      case 'openrouter': return 'badge-openrouter';
      case '1minai': return 'badge-1minai';
      case 'cloudflare': return 'badge-cloudflare';
      case 'sarvam': return 'badge-sarvam';
      case 'puter': return 'badge-puter';
      case 'agentrouter': return 'badge-agentrouter';
      case 'zenmux': return 'badge-zenmux';
      case 'gemini': return 'badge-gemini';
      case 'custom': return 'badge-custom';
      default: return 'badge-default';
    }
  }

  function connectAdminKey() {
    const key = appState.adminKey.trim();
    if (!key) return;
    localStorage.setItem('cag_admin_key', key);
    reloadCredentials();
    loadPools();
  }

  async function refreshAllProviders() {
    refreshLoading = true;
    appState.apiLoading = true;
    try {
      const res = await fetch('/api/v1/admin/providers/refresh', {
        method: 'POST',
        headers: adminHeaders()
      });
      if (res.ok) {
        const data = await res.json();
        appState.addToast('success', data.message || `Re-synced ${data.models_count ?? 0} model pools`);
        await reloadCredentials();
        appState.loadModels(true);
      } else {
        const err = await res.json();
        appState.addToast('error', err.details || err.error || 'Refresh failed');
      }
    } catch (e) {
      appState.addToast('error', `Network error: ${e.message}`);
    } finally {
      refreshLoading = false;
      appState.apiLoading = false;
    }
  }

  // ─── Re-Discovery Functions ────────────────────────────────────────────────
  async function checkRediscoveryStatus() {
    try {
      const res = await fetch('/api/v1/admin/providers/rediscover/status', {
        headers: adminHeaders()
      });
      if (res.ok) {
        rediscoveryStatus = await res.json();
        if (rediscoveryStatus.report) {
          try {
            parsedRediscoveryReport = typeof rediscoveryStatus.report === 'string'
              ? JSON.parse(rediscoveryStatus.report)
              : rediscoveryStatus.report;
          } catch (e) {
            parsedRediscoveryReport = null;
          }
        } else {
          parsedRediscoveryReport = null;
        }
      }
    } catch (e) {
      console.error('Failed to check re-discovery status', e);
    }
  }

  async function runReDiscovery() {
    isRediscovering = true;
    showRediscoveryDetails = false;
    parsedRediscoveryReport = null;
    try {
      const res = await fetch('/api/v1/admin/providers/rediscover', {
        method: 'POST',
        headers: adminHeaders()
      });
      if (res.ok) {
        rediscoveryStatus = { status: 'RUNNING' };
        // Start polling for status updates
        rediscoveryPollInterval = setInterval(async () => {
          await checkRediscoveryStatus();
          if (rediscoveryStatus.status !== 'RUNNING' && rediscoveryStatus.status !== 'PENDING') {
            clearInterval(rediscoveryPollInterval);
            rediscoveryPollInterval = null;
            isRediscovering = false;
            // Reload credentials to reflect any new models
            await reloadCredentials();
            appState.loadModels(true);
          }
        }, 2000);
      } else if (res.status === 409) {
        appState.addToast('warning', 'Re-discovery is already running. Please wait for it to complete.');
        isRediscovering = false;
      } else {
        const err = await res.json();
        appState.addToast('error', err.details || err.error || 'Failed to start re-discovery');
        isRediscovering = false;
      }
    } catch (e) {
      appState.addToast('error', `Network error: ${e.message}`);
      isRediscovering = false;
    }
  }

  function dismissRediscoveryBanner() {
    rediscoveryStatus = { status: 'IDLE' };
    parsedRediscoveryReport = null;
    showRediscoveryDetails = false;
  }

  function formatDuration(ms) {
    if (!ms) return '—';
    if (ms < 1000) return `${ms}ms`;
    const secs = Math.floor(ms / 1000);
    if (secs < 60) return `${secs}s`;
    return `${Math.floor(secs / 60)}m ${secs % 60}s`;
  }

  $effect(() => {
    const key = appState.getAdminKey();
    if (key && providerCredentials.length === 0 && !providerLoading) {
      reloadCredentials();
      loadPools();
      checkRediscoveryStatus();
    }
  });

  onMount(() => {
    if (appState.getAdminKey()) {
      reloadCredentials();
      loadPools();
      checkRediscoveryStatus();
    }
    return () => {
      if (rediscoveryPollInterval) clearInterval(rediscoveryPollInterval);
    };
  });
</script>

<header class="header flex items-center justify-between px-6 py-4 border-b shrink-0">
  <div class="flex items-center gap-3">
    <KeyRound size={20} class="text-[#f97316]" />
    <span class="font-bold text-base">Provider Credentials</span>
    {#if appState.adminKey.trim()}
      <span class="text-xs font-bold text-secondary bg-gray-500/10 border border-gray-500/20 px-2.5 py-0.5 rounded-full uppercase">{totalCount} registered</span>
    {/if}
  </div>
  
  {#if appState.adminKey.trim()}
    <div class="flex items-center gap-2 animate-fade-in">
      {#if selectedIds.length > 0}
        <Button variant="danger" size="sm" onclick={confirmBulkDelete} title="Delete selected credentials">
          <Trash2 size={14} />
          Delete Selected ({selectedIds.length})
        </Button>
      {/if}
      <Button variant="secondary" size="sm" onclick={refreshAllProviders} disabled={refreshLoading} title="Re-run discovery for all stored provider keys and provision any missing alias pools">
        <RefreshCw size={14} class={refreshLoading ? 'animate-spin' : ''} />
        {refreshLoading ? 'Refreshing...' : 'Refresh'}
      </Button>
      <Button
        variant="outline"
        size="sm"
        onclick={runReDiscovery}
        disabled={isRediscovering || rediscoveryStatus.status === 'RUNNING'}
        title="Scan all provider endpoints for newly available models and auto-register them"
      >
        <Radar size={14} class={isRediscovering || rediscoveryStatus.status === 'RUNNING' ? 'animate-spin' : ''} />
        <span>{isRediscovering || rediscoveryStatus.status === 'RUNNING' ? 'Scanning...' : 'Re-Discover Models'}</span>
      </Button>
      <Button variant="primary" size="sm" onclick={openAddProviderModal}>
        <Plus size={14} />
        Add Provider
      </Button>
    </div>
  {/if}
</header>

{#if !appState.adminKey.trim()}
  <!-- Admin key prompt -->
  <div class="logs-key-prompt flex flex-col justify-center items-center flex-grow p-6">
    <Card variant="filled" padding="lg" class="logs-key-card flex flex-col items-center text-center">
      <Shield size={40} class="text-[#f97316] mb-4 animate-pulse" />
      <h2 class="font-bold text-lg mb-2 text-primary">Admin Key Required</h2>
      <p class="text-sm mb-6 text-secondary max-w-sm">Enter your Admin API Key to manage provider credentials, pools, and API keys.</p>
      
      <div class="flex flex-col gap-3 w-full max-w-sm">
        <Input
          type="password"
          placeholder="Enter Admin API Key..."
          bind:value={appState.adminKey}
          onkeydown={(e) => { if (e.key === 'Enter') connectAdminKey(); }}
        />
        <Button variant="primary" size="md" onclick={connectAdminKey}>
          Connect
        </Button>
      </div>
      
      {#if providerError}
        <p class="text-red-500 text-sm font-semibold mt-4">{providerError}</p>
      {/if}
    </Card>
  </div>
{:else}
  <!-- Providers data grid (virtualized + lazy-loaded) -->
  <div class="providers-grid-wrap flex flex-col flex-grow overflow-hidden">

    <!-- ─── Re-Discovery Status Banner ──────────────────────────────────────── -->
    {#if rediscoveryStatus.status === 'RUNNING' || rediscoveryStatus.status === 'PENDING'}
      <div class="rediscovery-banner running">
        <div class="flex items-center gap-3">
          <div class="rediscovery-spinner"></div>
          <div>
            <p class="font-semibold text-sm">Scanning all provider endpoints for new models...</p>
            <p class="text-xs opacity-70 mt-0.5">This may take a minute depending on the number of providers.</p>
          </div>
        </div>
      </div>
    {:else if rediscoveryStatus.status === 'SUCCESS' && parsedRediscoveryReport}
      <div class="rediscovery-banner success">
        <div class="flex items-center justify-between w-full">
          <div class="flex items-center gap-2.5">
            <CheckCircle size={18} class="text-emerald-500 shrink-0" />
            <div>
              {#if parsedRediscoveryReport.new_models_added > 0}
                <span class="font-semibold text-sm">
                  Re-Discovery complete: Found and auto-registered {parsedRediscoveryReport.new_models_added} new model{parsedRediscoveryReport.new_models_added !== 1 ? 's' : ''}!
                </span>
              {:else}
                <span class="font-semibold text-sm">All models are up to date — no new models found.</span>
              {/if}
              <span class="text-xs opacity-60 ml-2">
                {parsedRediscoveryReport.successful_endpoints}/{parsedRediscoveryReport.total_endpoints_scanned} endpoints scanned
                · {formatDuration(parsedRediscoveryReport.duration_ms)}
              </span>
            </div>
          </div>
          <div class="flex items-center gap-2">
            {#if parsedRediscoveryReport.new_models_added > 0 || parsedRediscoveryReport.provider_breakdown?.length > 0}
              <button 
                onclick={() => showRediscoveryDetails = !showRediscoveryDetails}
                class="rediscovery-details-toggle"
              >
                {showRediscoveryDetails ? 'Hide Details' : 'View Details'}
                {#if showRediscoveryDetails}
                  <ChevronUp size={14} />
                {:else}
                  <ChevronDown size={14} />
                {/if}
              </button>
            {/if}
            <button onclick={dismissRediscoveryBanner} class="rediscovery-dismiss" title="Dismiss">
              <X size={14} />
            </button>
          </div>
        </div>

        {#if showRediscoveryDetails}
          <div class="rediscovery-details">
            {#if parsedRediscoveryReport.new_models?.length > 0}
              <div class="rediscovery-section">
                <h4 class="rediscovery-section-title">
                  <CheckCircle size={13} class="text-emerald-500" />
                  New Models Registered ({parsedRediscoveryReport.new_models.length})
                </h4>
                <div class="rediscovery-model-list">
                  {#each parsedRediscoveryReport.new_models as model}
                    <span class="rediscovery-model-tag new">{model}</span>
                  {/each}
                </div>
              </div>
            {/if}

            {#if parsedRediscoveryReport.provider_breakdown?.length > 0}
              <div class="rediscovery-section">
                <h4 class="rediscovery-section-title">
                  <Server size={13} />
                  Provider Breakdown
                </h4>
                <div class="rediscovery-provider-list">
                  {#each parsedRediscoveryReport.provider_breakdown as scan}
                    <div class="rediscovery-provider-row">
                      <div class="flex items-center gap-2">
                        {#if scan.status === 'success'}
                          <CheckCircle size={13} class="text-emerald-500 shrink-0" />
                        {:else}
                          <XCircle size={13} class="text-red-400 shrink-0" />
                        {/if}
                        <span class="provider-badge {providerBadgeClass(scan.provider)}" style="font-size: 10px;">{scan.provider}</span>
                        <span class="text-xs opacity-60 truncate" style="max-width: 200px;" title={scan.base_url}>{scan.base_url}</span>
                      </div>
                      <div class="flex items-center gap-3 text-xs">
                        {#if scan.status === 'success'}
                          <span class="opacity-70">{scan.models_synced} models synced</span>
                          {#if scan.new_models?.length > 0}
                            <span class="font-semibold text-emerald-600">+{scan.new_models.length} new</span>
                          {/if}
                        {:else}
                          <span class="text-red-400 truncate" style="max-width: 250px;" title={scan.error}>{scan.error}</span>
                        {/if}
                      </div>
                    </div>
                  {/each}
                </div>
              </div>
            {/if}

            {#if parsedRediscoveryReport.errors?.length > 0}
              <div class="rediscovery-section">
                <h4 class="rediscovery-section-title text-red-400">
                  <AlertTriangle size={13} />
                  Errors ({parsedRediscoveryReport.errors.length})
                </h4>
                <ul class="rediscovery-error-list">
                  {#each parsedRediscoveryReport.errors as err}
                    <li>{err}</li>
                  {/each}
                </ul>
              </div>
            {/if}
          </div>
        {/if}
      </div>
    {:else if rediscoveryStatus.status === 'FAILED'}
      <div class="rediscovery-banner error">
        <div class="flex items-center justify-between w-full">
          <div class="flex items-center gap-2.5">
            <XCircle size={18} class="text-red-400 shrink-0" />
            <div>
              <span class="font-semibold text-sm">Re-discovery scan failed</span>
              {#if rediscoveryStatus.error}
                <span class="text-xs opacity-70 ml-2">{rediscoveryStatus.error}</span>
              {/if}
            </div>
          </div>
          <button onclick={dismissRediscoveryBanner} class="rediscovery-dismiss" title="Dismiss">
            <X size={14} />
          </button>
        </div>
      </div>
    {/if}
    <!-- Filter / search toolbar -->
    <div class="flex flex-col border-b shrink-0 bg-[var(--card-bg)] shadow-sm">
      <div class="flex items-center gap-3 px-6 py-4 flex-wrap">
        <div class="relative flex items-center flex-grow max-w-md">
          <div class="absolute left-3.5 inset-y-0 flex items-center pointer-events-none z-10 text-secondary opacity-70">
            <Search size={16} />
          </div>
          <input
            type="text"
            class="filter-search-input"
            placeholder="Search provider, URL, model..."
            bind:value={searchQuery}
            oninput={onSearchInput}
          />
        </div>

        <select class="filter-select max-w-[200px]" bind:value={providerFilter} onchange={onSearchInput}>
          <option value="">All providers</option>
          <option value="openai">OpenAI</option>
          <option value="anthropic">Anthropic</option>
          <option value="nvidia">NVIDIA</option>
          <option value="ollama">Ollama</option>
          <option value="openrouter">OpenRouter</option>
          <option value="1minai">1min.ai</option>
          <option value="cloudflare">Cloudflare</option>
          <option value="sarvam">Sarvam AI</option>
          <option value="puter">Puter</option>
          <option value="zenmux">ZenMux</option>
          <option value="gemini">Gemini</option>
          <option value="custom">Custom</option>
        </select>

        {#if searchQuery || providerFilter}
          <Button 
            variant="ghost" 
            size="sm" 
            onclick={() => { searchQuery = ''; providerFilter = ''; onSearchInput(); }}
            class="text-xs text-red-500 hover:text-red-600 font-semibold px-3 h-[42px]"
          >
            Clear Filters
          </Button>
        {/if}

        <span class="text-xs text-secondary font-medium ml-auto">
          {#if totalCount > 0}
            Showing {providerCredentials.length} of {totalCount}
            {#if loadingMore}<span class="opacity-60"> · loading more…</span>{/if}
          {/if}
        </span>
      </div>
    </div>

    {#if providerLoading}
      <div class="providers-loading flex flex-col items-center justify-center flex-grow">
        <div class="animate-spin text-[#f97316] text-xl">⟳</div>
        <p class="text-sm mt-2 text-secondary">Loading credentials...</p>
      </div>
    {:else if providerError}
      <div class="providers-loading flex flex-col items-center justify-center flex-grow">
        <AlertTriangle size={40} class="text-red-500 mb-2" />
        <p class="text-red-500 text-sm font-semibold">{providerError}</p>
        <Button variant="primary" class="mt-4" onclick={reloadCredentials}>Retry</Button>
      </div>
    {:else if providerCredentials.length === 0}
      <div class="providers-loading flex flex-col items-center justify-center flex-grow">
        <Server size={48} class="opacity-20 mb-4" />
        <p class="opacity-50 text-sm text-secondary">No credentials registered yet.</p>
        <Button variant="primary" class="mt-4" onclick={openAddProviderModal}>
          <Plus size={14} /> Add First Provider
        </Button>
      </div>
    {:else}
      <!-- Virtualized table -->
      <div class="providers-table-container flex flex-col flex-grow overflow-hidden">
        <!-- Fixed header -->
        <div class="providers-table-header">
          <div class="providers-table-row providers-table-headrow">
            <div style="width: 40px; display: flex; align-items: center; justify-content: center;">
              <label class="ios-checkbox-wrapper" title="Select all credentials">
                <input
                  type="checkbox"
                  class="ios-checkbox-input"
                  checked={selectedIds.length === providerCredentials.length && providerCredentials.length > 0}
                  onchange={(e) => {
                    if (e.target.checked) {
                      selectedIds = providerCredentials.map(c => c.id);
                    } else {
                      selectedIds = [];
                    }
                  }}
                />
                <span class="ios-checkbox-box"></span>
              </label>
            </div>
            <div style="font-size: 11px;">ID</div>
            <div style="font-size: 11px;">Provider</div>
            <div style="font-size: 11px;">Model Pattern</div>
            <div style="font-size: 11px;">Base URL</div>
            <div style="font-size: 11px; text-align: center;">Weight</div>
            <div style="font-size: 11px; text-align: center;">Health</div>
            <div style="font-size: 11px;">Key</div>
            <div style="font-size: 11px; text-align: center;">Actions</div>
          </div>
        </div>
        <!-- Virtualized scroll body -->
        <div
          class="providers-table-body"
          bind:this={vscrollEl}
          bind:clientHeight={viewportHeight}
          onscroll={onVScroll}
        >
          <div style="height: {providerCredentials.length * ROW_HEIGHT}px; position: relative;">
            <div style="position: absolute; top: {visibleRange.padTop}px; left: 0; right: 0;">
              {#each visibleItems as cred, i (cred.id)}
                <div class="providers-table-row provider-row" style="height: {ROW_HEIGHT}px;">
                  <div style="width: 40px; display: flex; align-items: center; justify-content: center;">
                    <label class="ios-checkbox-wrapper">
                      <input
                        type="checkbox"
                        class="ios-checkbox-input"
                        value={cred.id}
                        bind:group={selectedIds}
                      />
                      <span class="ios-checkbox-box"></span>
                    </label>
                  </div>
                  <div class="font-mono text-xs opacity-60">#{cred.id}</div>
                  <div>
                    <span class="provider-badge {providerBadgeClass(cred.provider)}">{cred.provider}</span>
                  </div>
                  <div class="font-mono text-sm">{cred.model_pattern || '—'}</div>
                  <div class="text-sm truncate" style="max-width: 250px;" title={cred.base_url}>{cred.base_url}</div>
                  <div class="text-center font-mono text-sm">{cred.weight}</div>
                  <div class="text-center">
                    <span class="health-dot {cred.is_healthy ? 'healthy' : 'unhealthy'}" title={cred.is_healthy ? 'Healthy' : (cred.last_error || 'Unhealthy')}></span>
                  </div>
                  <div class="font-mono text-xs opacity-50">{cred.key_mask}</div>
                  <div>
                    <div class="flex items-center justify-center gap-1">
                      <Button variant="ghost" size="sm" onclick={() => openEditModal(cred)} title="Edit credential">
                        <Pencil size={15} />
                      </Button>
                      <Button variant="ghost" size="sm" onclick={() => confirmDelete(cred.id)} title="Delete credential">
                        <Trash2 size={15} class="text-red-500" />
                      </Button>
                    </div>
                  </div>
                </div>
              {/each}
            </div>
          </div>
          {#if loadingMore}
            <div class="flex items-center justify-center py-3 text-secondary text-sm">
              <span class="animate-spin mr-2">⟳</span> Loading more...
            </div>
          {/if}
        </div>
      </div>
    {/if}
  </div>
{/if}

<!-- ─── ADD PROVIDER MODAL ────────────────────────────────────────────────── -->
<Modal bind:show={showAddProviderModal}>
  {#snippet header()}
    <div class="flex border-b text-xs w-full">
      <button 
        class="tab-btn px-6 py-3 flex-grow font-semibold text-center {addProviderTab === 'standard' ? 'active' : ''}" 
        onclick={() => addProviderTab = 'standard'}
      >
        Standard Provider
      </button>
      <button 
        class="tab-btn px-6 py-3 flex-grow font-semibold text-center {addProviderTab === 'autodiscovery' ? 'active' : ''}" 
        onclick={() => addProviderTab = 'autodiscovery'}
      >
        Auto-Discovery
      </button>
    </div>
  {/snippet}

  {#if addProviderTab === 'standard'}
    <div class="flex flex-col gap-4">
      <p class="text-sm text-secondary leading-relaxed">Add a single credential to an existing model pool.</p>
      
      <Input type="select" label="Pool" bind:value={addProviderForm.pool_id} placeholder="Select a pool...">
        {#each providerPools as pool}
          <option value={pool.id}>{pool.model_pattern} (ID: {pool.id})</option>
        {/each}
      </Input>

      <Input type="select" label="Provider" bind:value={addProviderForm.provider}>
        <option value="openai">OpenAI</option>
        <option value="anthropic">Anthropic</option>
        <option value="nvidia">NVIDIA</option>
        <option value="ollama">Ollama</option>
        <option value="openrouter">OpenRouter</option>
        <option value="1minai">1min.ai</option>
        <option value="cloudflare">Cloudflare Workers AI</option>
        <option value="sarvam">Sarvam AI</option>
        <option value="puter">Puter.com</option>
        <option value="agentrouter">AgentRouter.org</option>
        <option value="zenmux">ZenMux</option>
        <option value="gemini">Google AI Studio (Gemini)</option>
        <option value="google">Google</option>
        <option value="custom">Custom</option>
      </Input>

      <Input type="password" label="API Key" placeholder="sk-..." bind:value={addProviderForm.api_key} />
      
      <Input type="text" label="Base URL" placeholder="https://api.openai.com" bind:value={addProviderForm.base_url} />
      
      <Input type="number" label="Weight" min="1" bind:value={addProviderForm.weight} />
    </div>
  {:else}
    <div class="flex flex-col gap-4">
      <p class="text-sm text-secondary leading-relaxed">Auto-discover models from NVIDIA NIM, Ollama Cloud, OpenRouter (free models only), or any OpenAI-compatible provider. Pools are created automatically.</p>
      
      <Input type="select" label="Provider Type" bind:value={autoDiscoverForm.provider} onchange={() => {
        if (autoDiscoverForm.provider === 'nvidia') {
          autoDiscoverForm.base_url = 'https://integrate.api.nvidia.com/v1';
        } else if (autoDiscoverForm.provider === 'ollama') {
          autoDiscoverForm.base_url = 'https://ollama.com';
        } else if (autoDiscoverForm.provider === 'openrouter') {
          autoDiscoverForm.base_url = 'https://openrouter.ai/api/v1';
        } else if (autoDiscoverForm.provider === '1minai') {
          autoDiscoverForm.base_url = 'https://api.1min.ai';
        } else if (autoDiscoverForm.provider === 'cloudflare') {
          autoDiscoverForm.base_url = '';
        } else if (autoDiscoverForm.provider === 'sarvam') {
          autoDiscoverForm.base_url = 'https://api.sarvam.ai';
        } else if (autoDiscoverForm.provider === 'puter') {
          autoDiscoverForm.base_url = 'https://api.puter.com/puterai/openai/v1';
        } else if (autoDiscoverForm.provider === 'agentrouter') {
          autoDiscoverForm.base_url = 'https://ps.air-outer.com/v1';
        } else if (autoDiscoverForm.provider === 'zenmux') {
          autoDiscoverForm.base_url = 'https://zenmux.ai/api/v1';
        } else if (autoDiscoverForm.provider === 'gemini') {
          autoDiscoverForm.base_url = 'https://generativelanguage.googleapis.com';
        } else {
          autoDiscoverForm.base_url = '';
        }
        autoDiscoverForm.label = '';
        autoDiscoverForm.account_id = '';
        autoDiscoverForm.api_token = '';
        autoDiscoverForm.api_key = '';
      }}>
        <option value="openrouter">OpenRouter (Free Models)</option>
        <option value="nvidia">NVIDIA NIM</option>
        <option value="ollama">Ollama Cloud</option>
        <option value="1minai">1min.ai (Multi-Modal)</option>
        <option value="cloudflare">Cloudflare Workers AI</option>
        <option value="sarvam">Sarvam AI</option>
        <option value="puter">Puter.com</option>
        <option value="agentrouter">AgentRouter.org</option>
        <option value="zenmux">ZenMux</option>
        <option value="gemini">Google AI Studio (Gemini)</option>
        <option value="custom">OpenAI-Compatible (Custom)</option>
      </Input>

      {#if autoDiscoverForm.provider === 'openrouter'}
        <div class="rounded-lg border border-indigo-500/20 bg-indigo-500/5 px-4 py-3 text-xs text-indigo-400 leading-relaxed">
          🆓 Only <strong>free-tier models</strong> will be registered (those with a <code>:free</code> identifier). No paid models will be added. Get your API key at <a href="https://openrouter.ai/settings/keys" target="_blank" rel="noopener noreferrer" class="underline">openrouter.ai/settings/keys</a>.
        </div>
      {/if}

      {#if autoDiscoverForm.provider === '1minai'}
        <div class="rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-4 py-3 text-xs text-emerald-400 leading-relaxed">
          🤖 <strong>1min.ai</strong> supports all modalities: Writing, Image, Audio, Video, and Code. All models are auto-discovered from a static manifest — enter your API key, or switch to <strong>Bulk / Batch Keys</strong> mode to register many keys at once (one per line; each key is validated and bound to every model pool). Get your key at <a href="https://app.1min.ai" target="_blank" rel="noopener noreferrer" class="underline">app.1min.ai</a>.
        </div>
      {/if}

      {#if autoDiscoverForm.provider === 'cloudflare'}
        <div class="rounded-lg border border-orange-500/20 bg-orange-500/5 px-4 py-3 text-xs text-orange-400 leading-relaxed">
          ☁️ <strong>Cloudflare Workers AI</strong> requires your <strong>Account ID</strong> and an <strong>API Token</strong> with Workers AI permissions. All available models (Text, Image, Audio, Embeddings) are discovered automatically. Find your Account ID and create an API token at <a href="https://dash.cloudflare.com" target="_blank" rel="noopener noreferrer" class="underline">dash.cloudflare.com</a>.
        </div>
      {/if}

      {#if autoDiscoverForm.provider === 'sarvam'}
        <div class="rounded-lg border border-purple-500/20 bg-purple-500/5 px-4 py-3 text-xs text-purple-400 leading-relaxed">
          🇮🇳 <strong>Sarvam AI</strong> is a premium AI provider in India. It offers a static set of chat models (<code>sarvam-30b</code>, <code>sarvam-105b</code>) and supports reasoning. Discovery is instant and uses a hardcoded manifest. Get your API subscription key at <a href="https://dashboard.sarvam.ai" target="_blank" rel="noopener noreferrer" class="underline">dashboard.sarvam.ai</a>.
        </div>
      {/if}

      {#if autoDiscoverForm.provider === 'puter'}
        <div class="rounded-lg border border-blue-500/20 bg-blue-500/5 px-4 py-3 text-xs text-blue-400 leading-relaxed">
          🚀 <strong>Puter.com</strong> is a developer-friendly cloud with free AI access. Enter your API Token (Puter Auth Token) to auto-discover all models — or switch to <strong>Bulk / Batch Keys</strong> mode to register many tokens at once (one per line). Get your API Token at <a href="https://puter.com/dashboard" target="_blank" rel="noopener noreferrer" class="underline">puter.com/dashboard</a>.
        </div>
      {/if}

      {#if autoDiscoverForm.provider === 'agentrouter'}
        <div class="rounded-lg border border-teal-500/20 bg-teal-500/5 px-4 py-3 text-xs text-teal-400 leading-relaxed">
          🌐 <strong>AgentRouter.org</strong> is an AI API aggregator. Enter your API Key to auto-discover all available models. Standard OpenAI requests are automatically sanitized upstream (Issue #1921 fix). Get your API key at <a href="https://agentrouter.org" target="_blank" rel="noopener noreferrer" class="underline">agentrouter.org</a>.
        </div>
      {/if}

      {#if autoDiscoverForm.provider === 'zenmux'}
        <div class="rounded-lg border border-purple-500/20 bg-purple-500/5 px-4 py-3 text-xs text-purple-400 leading-relaxed">
          🌐 <strong>ZenMux</strong> is an enterprise-grade AI model aggregator. Enter your API Key to auto-discover all aggregated models. Get your key at <a href="https://zenmux.ai" target="_blank" rel="noopener noreferrer" class="underline">zenmux.ai</a>.
        </div>
      {/if}

      {#if autoDiscoverForm.provider === 'gemini'}
        <div class="rounded-lg border border-cyan-500/20 bg-cyan-500/5 px-4 py-3 text-xs text-cyan-400 leading-relaxed">
          ✨ <strong>Google AI Studio (Gemini)</strong> provides high-performance reasoning and multimodal models. Enter your Gemini API key (starts with <code>AIzaSy</code>) to discover and register all available models under both <code>gemini/&lt;model_id&gt;</code> and clean aliases. Get your API key at <a href="https://aistudio.google.com" target="_blank" rel="noopener noreferrer" class="underline">aistudio.google.com</a>.
        </div>
      {/if}

      {#if autoDiscoverForm.provider === 'custom'}
        <Input type="text" label="Label (namespace prefix)" placeholder="e.g. huggingface, together, deepinfra" bind:value={autoDiscoverForm.label} />
      {/if}

      {#if autoDiscoverForm.provider === 'custom' || autoDiscoverForm.provider === 'puter' || autoDiscoverForm.provider === '1minai'}
        <!-- Key Mode Toggle (Single vs Batch) -->
        <div class="flex flex-col gap-1.5">
          <div class="flex items-center justify-between">
            <span class="text-xs font-bold uppercase tracking-wider text-secondary">Key Registration Mode</span>
            <div class="inline-flex rounded-lg bg-[var(--frame-bg)] p-0.5 border border-[var(--border-color)]">
              <button
                type="button"
                class="flex items-center gap-1.5 px-3 py-1 text-xs font-semibold rounded-md transition-all {customKeyMode === 'single' ? 'bg-[#f97316] text-white shadow-sm' : 'text-secondary hover:text-[var(--text-primary)]'}"
                onclick={() => customKeyMode = 'single'}
              >
                <KeyRound size={12} />
                <span>Single Key</span>
              </button>
              <button
                type="button"
                class="flex items-center gap-1.5 px-3 py-1 text-xs font-semibold rounded-md transition-all {customKeyMode === 'batch' ? 'bg-[#f97316] text-white shadow-sm' : 'text-secondary hover:text-[var(--text-primary)]'}"
                onclick={() => customKeyMode = 'batch'}
              >
                <Layers size={12} />
                <span>Bulk / Batch Keys</span>
                {#if parsedBatchKeysCount > 0}
                  <span class="ml-0.5 px-1.5 py-0.2 text-[10px] font-bold rounded-full {customKeyMode === 'batch' ? 'bg-white/20 text-white' : 'bg-orange-500/20 text-orange-400'}">
                    {parsedBatchKeysCount}
                  </span>
                {/if}
              </button>
            </div>
          </div>
        </div>

        {#if customKeyMode === 'batch'}
          <div class="flex flex-col gap-1.5">
            <div class="flex items-center justify-between">
              <span class="text-xs font-bold uppercase tracking-wider text-secondary">API Keys (Bulk)</span>
              {#if parsedBatchKeysCount > 0}
                <span class="text-xs font-medium text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded-full flex items-center gap-1">
                  <CheckCircle size={11} /> {parsedBatchKeysCount} {parsedBatchKeysCount === 1 ? 'key' : 'keys'} detected
                </span>
              {/if}
            </div>
            <Input
              type="textarea"
              rows={5}
              placeholder="Paste multiple API keys here (one key per line, or comma-separated)...&#10;sk-key-1...&#10;sk-key-2...&#10;sk-key-3..."
              bind:value={batchKeysText}
            />
            <p class="text-[11px] text-secondary/70">Each key will be auto-discovered. Successful keys will bind to the model pools for round-robin load balancing.</p>
          </div>
        {:else if autoDiscoverForm.provider === 'custom'}
          <Input 
            type="password" 
            label="API Key" 
            placeholder="Bearer API key..." 
            bind:value={autoDiscoverForm.api_key} 
          />
        {/if}
      {/if}

      {#if autoDiscoverForm.provider === 'cloudflare'}
        <!-- Cloudflare needs Account ID + API Token instead of the generic api_key/base_url -->
        <Input
          type="text"
          label="Account ID"
          placeholder="a1b2c3d4e5f6..."
          bind:value={autoDiscoverForm.account_id}
        />
        <Input
          type="password"
          label="API Token"
          placeholder="Workers AI API Token..."
          bind:value={autoDiscoverForm.api_token}
        />
      {:else if autoDiscoverForm.provider !== 'custom' && !(autoDiscoverForm.provider === 'puter' && customKeyMode === 'batch') && !(autoDiscoverForm.provider === '1minai' && customKeyMode === 'batch')}
        <Input 
          type="password" 
          label="API Key" 
          placeholder={
            autoDiscoverForm.provider === 'nvidia' ? 'nvapi-...' :
            autoDiscoverForm.provider === 'ollama' ? 'Ollama Cloud API key...' :
            autoDiscoverForm.provider === 'openrouter' ? 'sk-or-v1-...' :
            autoDiscoverForm.provider === '1minai' ? '1min.ai API key...' :
            autoDiscoverForm.provider === 'sarvam' ? 'Sarvam API key (api-subscription-key)...' :
            autoDiscoverForm.provider === 'puter' ? 'Puter Auth Token...' :
            autoDiscoverForm.provider === 'agentrouter' ? 'AgentRouter API key (sk-...)...' :
            autoDiscoverForm.provider === 'zenmux' ? 'ZenMux API Key...' :
            autoDiscoverForm.provider === 'gemini' ? 'Google AI Studio API Key (AIzaSy...)...' :
            'Bearer API key...'
          } 
          bind:value={autoDiscoverForm.api_key} 
        />
      {/if}

      {#if autoDiscoverForm.provider !== 'openrouter' && autoDiscoverForm.provider !== '1minai' && autoDiscoverForm.provider !== 'sarvam' && autoDiscoverForm.provider !== 'puter' && autoDiscoverForm.provider !== 'agentrouter' && autoDiscoverForm.provider !== 'zenmux' && autoDiscoverForm.provider !== 'gemini' && autoDiscoverForm.provider !== 'cloudflare'}
        <Input type="text" label="Base URL" placeholder={autoDiscoverForm.provider === 'custom' ? 'https://api.together.xyz/v1' : ''} bind:value={autoDiscoverForm.base_url} />
      {/if}
      
      <Input type="number" label="Weight" min="1" bind:value={autoDiscoverForm.weight} />

      {#if autoDiscoverLoading}
        <div class="rounded-xl border border-orange-500/20 bg-orange-500/5 p-4 flex flex-col gap-2.5">
          <div class="flex items-center gap-3">
            <RefreshCw size={16} class="animate-spin text-orange-400 shrink-0" />
            <div class="flex flex-col">
              <span class="text-xs font-semibold text-orange-400">
                {#if (autoDiscoverForm.provider === 'custom' || autoDiscoverForm.provider === 'puter' || autoDiscoverForm.provider === '1minai') && customKeyMode === 'batch' && parsedBatchKeysCount > 0}
                  Auto-discovering and registering {parsedBatchKeysCount} keys...
                {:else}
                  Validating credentials and synchronizing models...
                {/if}
              </span>
              <span class="text-[11px] text-secondary">
                Querying provider endpoint and binding credentials to model pools
              </span>
            </div>
          </div>
        </div>
      {/if}

      {#if batchReport}
        <div class="rounded-xl border {batchReport.failed_count > 0 ? 'border-amber-500/30 bg-amber-500/5' : 'border-emerald-500/30 bg-emerald-500/5'} p-4 flex flex-col gap-3">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-2">
              {#if batchReport.failed_count > 0}
                <AlertTriangle size={16} class="text-amber-400 shrink-0" />
                <span class="text-xs font-bold text-amber-400">Batch Processing Summary</span>
              {:else}
                <CheckCircle size={16} class="text-emerald-400 shrink-0" />
                <span class="text-xs font-bold text-emerald-400">Batch Discovery Completed</span>
              {/if}
            </div>
            <span class="text-xs font-medium text-secondary">
              {batchReport.success_count}/{batchReport.total_keys} Succeeded
            </span>
          </div>

          <div class="flex flex-col gap-1.5 max-h-48 overflow-y-auto pr-1">
            {#each batchReport.results as res}
              <div class="flex items-center justify-between text-xs px-3 py-2 rounded-lg bg-[var(--frame-bg)] border border-[var(--border-color)]">
                <div class="flex items-center gap-2 font-mono">
                  {#if res.success}
                    <CheckCircle size={13} class="text-emerald-400 shrink-0" />
                  {:else}
                    <XCircle size={13} class="text-red-400 shrink-0" />
                  {/if}
                  <span class="text-[var(--text-primary)]">Key #{res.index}: {res.key_masked}</span>
                </div>
                {#if res.success}
                  <span class="text-emerald-400 font-semibold">{res.models_count} models</span>
                {:else}
                  <span class="text-red-400 truncate max-w-[200px]" title={res.error}>{res.error || 'Failed'}</span>
                {/if}
              </div>
            {/each}
          </div>

          {#if batchReport.failed_count > 0}
            <div class="flex justify-end pt-1">
              <Button size="sm" variant="outline" onclick={() => { batchReport = null; }}>Dismiss</Button>
            </div>
          {/if}
        </div>
      {/if}
    </div>
  {/if}

  {#snippet footer()}
    <div class="flex justify-end gap-3 w-full">
      <Button variant="outline" onclick={() => showAddProviderModal = false}>Cancel</Button>
      {#if addProviderTab === 'standard'}
        <Button variant="primary" onclick={createCredential} disabled={addProviderLoading || !addProviderForm.pool_id}>
          {#if addProviderLoading}
            <span class="animate-spin">⟳</span> Creating...
          {:else}
            Create Credential
          {/if}
        </Button>
      {:else}
        {#if batchReport && batchReport.failed_count > 0}
          <Button variant="primary" onclick={() => { showAddProviderModal = false; batchReport = null; }}>
            Done
          </Button>
        {:else}
          <Button variant="primary" onclick={autoDiscoverProvider} disabled={autoDiscoverLoading}>
            {#if autoDiscoverLoading}
              <span class="animate-spin">⟳</span> Discovering...
            {:else}
              Discover & Register {(autoDiscoverForm.provider === 'custom' || autoDiscoverForm.provider === 'puter' || autoDiscoverForm.provider === '1minai') && customKeyMode === 'batch' && parsedBatchKeysCount > 0 ? `(${parsedBatchKeysCount} Keys)` : ''}
            {/if}
          </Button>
        {/if}
      {/if}
    </div>
  {/snippet}
</Modal>

<!-- ─── EDIT CREDENTIAL MODAL ──────────────────────────────────────────────── -->
<Modal bind:show={showEditModal} title="Edit Credential">
  <div class="flex flex-col gap-4">
    <Input type="select" label="Provider" bind:value={editForm.provider}>
      <option value="openai">OpenAI</option>
      <option value="anthropic">Anthropic</option>
      <option value="nvidia">NVIDIA</option>
      <option value="ollama">Ollama</option>
      <option value="openrouter">OpenRouter</option>
      <option value="1minai">1min.ai</option>
      <option value="cloudflare">Cloudflare Workers AI</option>
      <option value="sarvam">Sarvam AI</option>
      <option value="puter">Puter.com</option>
      <option value="agentrouter">AgentRouter.org</option>
      <option value="zenmux">ZenMux</option>
      <option value="gemini">Google AI Studio (Gemini)</option>
      <option value="google">Google</option>
      <option value="custom">Custom</option>
    </Input>

    <Input type="password" label="New API Key" placeholder="Leave blank to keep current key" bind:value={editForm.api_key} />
    
    <Input type="text" label="Base URL" bind:value={editForm.base_url} />
    
    <div class="flex gap-4 items-end">
      <div class="flex-grow">
        <Input type="number" label="Weight" min="1" bind:value={editForm.weight} />
      </div>
      <div class="flex flex-col gap-2 shrink-0">
        <span class="text-xs font-bold uppercase tracking-wider text-secondary">Healthy</span>
        <label class="toggle-switch" style="margin-bottom: 9px;">
          <input type="checkbox" bind:checked={editForm.is_healthy} />
          <span class="toggle-slider"></span>
        </label>
      </div>
    </div>
  </div>

  {#snippet footer()}
    <div class="flex justify-end gap-3 w-full">
      <Button variant="outline" onclick={() => showEditModal = false}>Cancel</Button>
      <Button variant="primary" onclick={updateCredential} disabled={editLoading}>
        {#if editLoading}
          <span class="animate-spin">⟳</span> Saving...
        {:else}
          Save Changes
        {/if}
      </Button>
    </div>
  {/snippet}
</Modal>

<!-- ─── DELETE CONFIRMATION DIALOG ─────────────────────────────────────────── -->
<Modal bind:show={showDeleteConfirm} title="Delete Credential?">
  <div class="flex flex-col items-center gap-4 text-center">
    <AlertTriangle size={48} class="text-red-500 mb-2" />
    <p class="text-sm text-secondary">
      This credential will be removed from gateway routing immediately. Traffic will be routed to other healthy keys in the pool.
    </p>
    <p class="text-xs text-red-500 font-bold">This action is permanent and cannot be undone.</p>
  </div>

  {#snippet footer()}
    <div class="flex justify-center gap-3 w-full">
      <Button variant="outline" onclick={() => { showDeleteConfirm = false; deleteTargetId = null; }}>Cancel</Button>
      <Button variant="danger" onclick={deleteCredentialById} disabled={deleteLoading}>
        {#if deleteLoading}
          <span class="animate-spin">⟳</span>
        {:else}
          Delete
        {/if}
      </Button>
    </div>
  {/snippet}
</Modal>

<!-- ─── BULK DELETE CONFIRMATION DIALOG ─────────────────────────────────────────── -->
<Modal bind:show={showBulkDeleteConfirm} title="Delete Credentials?">
  <div class="flex flex-col items-center gap-4 text-center">
    <AlertTriangle size={48} class="text-red-500 mb-2" />
    <p class="text-sm text-secondary">
      Are you sure you want to permanently delete the {selectedIds.length} selected credentials?
      They will be removed from gateway routing immediately.
    </p>
    <p class="text-xs text-red-500 font-bold">This action is permanent and cannot be undone.</p>
  </div>

  {#snippet footer()}
    <div class="flex justify-center gap-3 w-full">
      <Button variant="outline" onclick={() => { showBulkDeleteConfirm = false; }}>Cancel</Button>
      <Button variant="danger" onclick={deleteCredentialsBulk} disabled={bulkDeleteLoading}>
        {#if bulkDeleteLoading}
          <span class="animate-spin">⟳</span>
        {:else}
          Delete ({selectedIds.length})
        {/if}
      </Button>
    </div>
  {/snippet}
</Modal>

<style>
  .tab-btn {
    border: none;
    color: var(--text-secondary);
    background: transparent;
    transition: all 0.2s;
    font-weight: 600;
    cursor: pointer;
    border-bottom: 2px solid transparent;
  }
  .tab-btn.active {
    color: #f97316;
    border-bottom: 2px solid #f97316;
  }

  /* ─── Virtualized table ─── */
  .providers-table-header {
    flex-shrink: 0;
    overflow-x: auto;
    background-color: var(--card-bg);
    border-bottom: 2px solid var(--border-color);
  }
  .providers-table-body {
    flex-grow: 1;
    overflow-y: auto;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }
  .providers-table-row {
    display: grid;
    grid-template-columns: 40px 70px 130px 1fr 1.4fr 70px 70px 130px 110px;
    align-items: center;
    min-width: 900px;
  }
  .providers-table-headrow {
    padding: 0;
  }
  .providers-table-headrow > div {
    padding: 12px 16px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 700;
    color: var(--text-secondary);
    white-space: nowrap;
  }
  .providers-table-row.provider-row {
    border-bottom: 1px solid var(--border-color);
    transition: background-color 0.15s;
    background-color: var(--card-bg);
  }
  .providers-table-row.provider-row:hover {
    background-color: var(--item-hover);
  }
  .providers-table-row.provider-row > div {
    padding: 12px 16px;
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  /* OpenRouter brand badge — indigo tone matching openrouter.ai visual identity */
  :global(.badge-openrouter) {
    background: rgba(99, 102, 241, 0.12);
    color: #818cf8;
    border: 1px solid rgba(99, 102, 241, 0.25);
  }

  /* 1min.ai badge — emerald tone */
  :global(.badge-1minai) {
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.25);
  }

  /* Cloudflare brand badge — orange/amber tone matching Cloudflare visual identity */
  :global(.badge-cloudflare) {
    background: rgba(249, 115, 22, 0.12);
    color: #fb923c;
    border: 1px solid rgba(249, 115, 22, 0.25);
  }

  /* Sarvam AI brand badge — purple/violet tone */
  :global(.badge-sarvam) {
    background: rgba(167, 139, 250, 0.12);
    color: #a78bfa;
    border: 1px solid rgba(167, 139, 250, 0.25);
  }

  /* Puter brand badge — blue/indigo tone */
  :global(.badge-puter) {
    background: rgba(59, 130, 246, 0.12);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.25);
  }

  /* AgentRouter brand badge — cyan/teal tone */
  :global(.badge-agentrouter) {
    background: rgba(20, 184, 166, 0.12);
    color: #2dd4bf;
    border: 1px solid rgba(20, 184, 166, 0.25);
  }

  /* ZenMux brand badge — purple/violet tone */
  :global(.badge-zenmux) {
    background: rgba(167, 139, 250, 0.12);
    color: #a78bfa;
    border: 1px solid rgba(167, 139, 250, 0.25);
  }

  /* Gemini brand badge — cyan/blue tone matching Google AI visual identity */
  :global(.badge-gemini) {
    background: rgba(6, 182, 212, 0.12);
    color: #22d3ee;
    border: 1px solid rgba(6, 182, 212, 0.25);
  }

  /* ─── Re-Discovery Button ─── */
  .rediscover-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    font-size: 13px;
    font-weight: 600;
    color: #fff;
    background: linear-gradient(135deg, #f97316 0%, #ea580c 50%, #dc2626 100%);
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(249, 115, 22, 0.3);
    white-space: nowrap;
  }
  .rediscover-btn:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(249, 115, 22, 0.45);
    filter: brightness(1.08);
  }
  .rediscover-btn:active:not(:disabled) {
    transform: translateY(0);
  }
  .rediscover-btn:disabled {
    opacity: 0.55;
    cursor: not-allowed;
    box-shadow: none;
  }

  /* ─── Re-Discovery Status Banner ─── */
  .rediscovery-banner {
    padding: 12px 20px;
    border-bottom: 1px solid var(--border-color);
    flex-shrink: 0;
    animation: bannerSlideIn 0.3s ease;
  }
  .rediscovery-banner.running {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(99, 102, 241, 0.06) 100%);
    border-left: 3px solid #3b82f6;
  }
  .rediscovery-banner.success {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.06) 0%, rgba(5, 150, 105, 0.04) 100%);
    border-left: 3px solid #10b981;
  }
  .rediscovery-banner.error {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.06) 0%, rgba(220, 38, 38, 0.04) 100%);
    border-left: 3px solid #ef4444;
  }

  @keyframes bannerSlideIn {
    from { opacity: 0; transform: translateY(-8px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* Spinner for the running state */
  .rediscovery-spinner {
    width: 18px;
    height: 18px;
    border: 2.5px solid rgba(59, 130, 246, 0.2);
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    flex-shrink: 0;
  }
  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  /* View Details toggle */
  .rediscovery-details-toggle {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 600;
    color: #10b981;
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }
  .rediscovery-details-toggle:hover {
    background: rgba(16, 185, 129, 0.14);
    border-color: rgba(16, 185, 129, 0.35);
  }

  /* Dismiss button */
  .rediscovery-dismiss {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border: none;
    background: transparent;
    border-radius: 6px;
    cursor: pointer;
    color: var(--text-secondary);
    opacity: 0.5;
    transition: all 0.15s;
  }
  .rediscovery-dismiss:hover {
    opacity: 1;
    background: rgba(0,0,0,0.06);
  }

  /* Expandable details panel */
  .rediscovery-details {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid rgba(16, 185, 129, 0.15);
    max-height: 350px;
    overflow-y: auto;
    animation: detailsExpand 0.25s ease;
  }
  @keyframes detailsExpand {
    from { opacity: 0; max-height: 0; }
    to   { opacity: 1; max-height: 350px; }
  }

  .rediscovery-section {
    margin-bottom: 12px;
  }
  .rediscovery-section:last-child {
    margin-bottom: 0;
  }
  .rediscovery-section-title {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    margin-bottom: 8px;
  }

  /* New model tags */
  .rediscovery-model-list {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
  }
  .rediscovery-model-tag {
    display: inline-block;
    padding: 3px 8px;
    font-size: 11px;
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-weight: 500;
    border-radius: 5px;
    white-space: nowrap;
  }
  .rediscovery-model-tag.new {
    background: rgba(16, 185, 129, 0.1);
    color: #059669;
    border: 1px solid rgba(16, 185, 129, 0.25);
  }

  /* Provider breakdown list */
  .rediscovery-provider-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .rediscovery-provider-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 10px;
    border-radius: 6px;
    background: rgba(0, 0, 0, 0.02);
    border: 1px solid var(--border-color);
    transition: background 0.15s;
  }
  .rediscovery-provider-row:hover {
    background: rgba(0, 0, 0, 0.04);
  }

  /* Error list */
  .rediscovery-error-list {
    list-style: none;
    padding: 0;
    margin: 0;
  }
  .rediscovery-error-list li {
    font-size: 11px;
    font-family: 'SF Mono', 'Fira Code', monospace;
    color: #f87171;
    padding: 4px 8px;
    border-left: 2px solid rgba(239, 68, 68, 0.3);
    margin-bottom: 3px;
    word-break: break-word;
  }
</style>
