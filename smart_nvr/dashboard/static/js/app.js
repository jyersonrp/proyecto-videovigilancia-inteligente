/**
 * Smart NVR — Core Application, Router, API Client & Status Polling
 */

const App = {
  activeTab: 'live',
  statusInterval: null,

  async init() {
    this.setupNavigation();
    this.startStatusPolling();

    // Initialize individual view components
    if (window.LiveGrid) LiveGrid.init();
    if (window.EventsGallery) EventsGallery.init();
    if (window.RoiEditor) RoiEditor.init();
    if (window.SettingsView) SettingsView.init();

    // Default tab
    this.switchTab('live');
  },

  setupNavigation() {
    // Desktop Tabs
    document.querySelectorAll('#nav-tabs .tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const tab = btn.getAttribute('data-tab');
        this.switchTab(tab);
      });
    });

    // Mobile Tabs
    document.querySelectorAll('.tab-btn-m').forEach(btn => {
      btn.addEventListener('click', () => {
        const tab = btn.getAttribute('data-tab');
        this.switchTab(tab);
      });
    });
  },

  switchTab(tabName) {
    this.activeTab = tabName;

    // Update Tab Button Styles
    document.querySelectorAll('#nav-tabs .tab-btn').forEach(btn => {
      const match = btn.getAttribute('data-tab') === tabName;
      if (match) {
        btn.className = 'tab-btn active px-4 py-2 rounded-md text-sm font-medium transition-colors bg-blue-600 text-white shadow';
      } else {
        btn.className = 'tab-btn px-4 py-2 rounded-md text-sm font-medium text-slate-300 hover:text-white hover:bg-slate-800 transition-colors';
      }
    });

    // Update Mobile Tab Styles
    document.querySelectorAll('.tab-btn-m').forEach(btn => {
      const match = btn.getAttribute('data-tab') === tabName;
      btn.className = match ? 'tab-btn-m px-3 py-1.5 text-xs font-semibold rounded text-blue-400' : 'tab-btn-m px-3 py-1.5 text-xs font-semibold rounded text-slate-400';
    });

    // Toggle Section Visibility
    const views = ['live', 'events', 'roi', 'settings'];
    views.forEach(v => {
      const sec = document.getElementById(`view-${v}`);
      if (sec) {
        if (v === tabName) {
          sec.classList.remove('hidden');
        } else {
          sec.classList.add('hidden');
        }
      }
    });

    // Notify components of activation / deactivation
    if (tabName === 'live') {
      if (window.LiveGrid) LiveGrid.onActivate();
    } else {
      if (window.LiveGrid) LiveGrid.onDeactivate();
    }
    if (tabName === 'events' && window.EventsGallery) EventsGallery.onActivate();
    if (tabName === 'roi' && window.RoiEditor) RoiEditor.onActivate();
    if (tabName === 'settings' && window.SettingsView) SettingsView.onActivate();
  },

  startStatusPolling() {
    this.pollHealth();
    this.statusInterval = setInterval(() => this.pollHealth(), 5000);
  },

  async pollHealth() {
    try {
      const res = await fetch('/api/health');
      if (!res.ok) return;
      const data = await res.json();

      const elCams = document.getElementById('header-active-cams');
      if (elCams) elCams.textContent = `${data.active_cameras} / ${data.total_cameras} Cámaras activas`;

      const elStorage = document.getElementById('header-storage-used');
      if (elStorage) elStorage.textContent = `${data.storage_used_mb.toFixed(1)} MB`;

      const elUptime = document.getElementById('header-uptime');
      if (elUptime) {
        const secs = Math.floor(data.uptime_seconds);
        const m = Math.floor(secs / 60);
        const s = secs % 60;
        elUptime.textContent = `${m}m ${s}s`;
      }
    } catch (e) {
      console.debug('Health poll failed', e);
    }
  },

  showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    const bgColors = {
      success: 'bg-emerald-800/90 border-emerald-600 text-emerald-100',
      error: 'bg-red-800/90 border-red-600 text-red-100',
      warning: 'bg-amber-800/90 border-amber-600 text-amber-100',
      info: 'bg-slate-800/90 border-slate-700 text-slate-100',
    };

    toast.className = `p-3 rounded-lg border shadow-lg text-xs font-medium backdrop-blur-sm pointer-events-auto transition-all duration-300 transform translate-y-2 opacity-0 ${bgColors[type] || bgColors.info}`;
    toast.textContent = message;

    container.appendChild(toast);
    setTimeout(() => {
      toast.classList.remove('translate-y-2', 'opacity-0');
    }, 10);

    setTimeout(() => {
      toast.classList.add('opacity-0', 'translate-y-2');
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }
};

window.addEventListener('DOMContentLoaded', () => App.init());
