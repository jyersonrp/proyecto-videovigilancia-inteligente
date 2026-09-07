/**
 * Smart NVR — Live Multi-Stream Grid View & Telemetry
 */

const LiveGrid = {
  cameras: [],
  pollTimer: null,
  activeLayout: 'auto',

  init() {
    this.setupControls();
    this.fetchCameras();
  },

  onActivate() {
    this.fetchCameras();
    if (!this.pollTimer) {
      this.pollTimer = setInterval(() => this.updateTelemetry(), 2500);
    }
  },

  onDeactivate() {
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
    // Release active MJPEG HTTP connections to free browser connection pool
    document.querySelectorAll('#live-grid .video-stream-img').forEach(img => {
      img.src = '';
    });
  },

  setupControls() {
    const selectLayout = document.getElementById('grid-layout-select');
    if (selectLayout) {
      selectLayout.addEventListener('change', (e) => {
        this.activeLayout = e.target.value;
        this.applyGridLayout();
      });
    }

    const btnRefresh = document.getElementById('btn-refresh-grid');
    if (btnRefresh) {
      btnRefresh.addEventListener('click', () => {
        this.fetchCameras();
        App.showToast('Actualizando feeds en vivo...', 'info');
      });
    }
  },

  applyGridLayout() {
    const grid = document.getElementById('live-grid');
    if (!grid) return;

    grid.className = 'grid gap-6';
    if (this.activeLayout === '1') {
      grid.classList.add('grid-cols-1');
    } else if (this.activeLayout === '2') {
      grid.classList.add('grid-cols-1', 'md:grid-cols-2');
    } else if (this.activeLayout === '3') {
      grid.classList.add('grid-cols-1', 'md:grid-cols-2', 'lg:grid-cols-3');
    } else {
      // Auto layout based on camera count
      const count = this.cameras.length;
      if (count <= 1) {
        grid.classList.add('grid-cols-1');
      } else if (count <= 4) {
        grid.classList.add('grid-cols-1', 'md:grid-cols-2');
      } else {
        grid.classList.add('grid-cols-1', 'md:grid-cols-2', 'lg:grid-cols-3');
      }
    }
  },

  async fetchCameras() {
    try {
      const res = await fetch('/api/cameras');
      if (!res.ok) return;
      this.cameras = await res.json();
      this.renderGrid();
      this.applyGridLayout();
    } catch (e) {
      console.error('Failed to load cameras for live grid', e);
    }
  },

  renderGrid() {
    const grid = document.getElementById('live-grid');
    if (!grid) return;

    if (this.cameras.length === 0) {
      grid.innerHTML = `
        <div class="col-span-full py-16 text-center text-slate-500 text-xs">
          No hay cámaras configuradas. Agregue una en la pestaña de Configuración.
        </div>
      `;
      return;
    }

    grid.innerHTML = this.cameras.map(cam => {
      const isEnabled = Boolean(cam.enabled);
      const isOnline = Boolean(cam.is_running);
      const statusClass = isOnline ? 'status-online' : 'status-offline';
      const statusText = isOnline ? 'ONLINE' : (isEnabled ? 'OFFLINE' : 'PAUSADA');
      const alertPulseClass = cam.alert_active ? 'alert-active' : (cam.motion_detected ? 'motion-active' : '');

      return `
        <div id="cam-card-${cam.id}" class="camera-card ${alertPulseClass} bg-slate-900 border border-slate-800 rounded-lg overflow-hidden shadow flex flex-col transition-all duration-300">
          <!-- Card Header -->
          <div class="px-4 py-2.5 bg-slate-950/80 border-b border-slate-800/80 flex items-center justify-between text-xs">
            <div class="flex items-center space-x-2">
              <span class="font-bold text-slate-100">${cam.name}</span>
              <span class="status-pill ${statusClass}">${statusText}</span>
              <button onclick="LiveGrid.toggleCamera('${cam.id}', ${!isEnabled})" 
                      class="px-2 py-0.5 rounded text-[10px] font-semibold border transition flex items-center space-x-1 ${isEnabled ? 'bg-emerald-950/80 border-emerald-700/70 text-emerald-300 hover:bg-emerald-900/80' : 'bg-amber-950/80 border-amber-700/70 text-amber-300 hover:bg-amber-900/80'}"
                      title="${isEnabled ? 'Pausar/desactivar esta cámara' : 'Activar esta cámara'}">
                <span class="w-1.5 h-1.5 rounded-full ${isEnabled ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}"></span>
                <span>${isEnabled ? 'Activa' : 'Pausada'}</span>
              </button>
            </div>
            <div class="flex items-center space-x-2">
              <span id="eco-badge-${cam.id}" class="status-pill bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 ${cam.eco_mode ? '' : 'hidden'}">🌿 ECO</span>
              <span id="fps-badge-${cam.id}" class="status-pill bg-slate-800 text-blue-400 font-mono">${cam.current_fps ? cam.current_fps.toFixed(1) : cam.fps_target} FPS</span>
              <span id="motion-badge-${cam.id}" class="status-pill bg-amber-500/20 text-amber-400 border border-amber-500/40 ${cam.motion_detected ? '' : 'hidden'}">Movimiento</span>
              <span id="alert-badge-${cam.id}" class="status-pill bg-red-500/20 text-red-400 border border-red-500/40 ${cam.alert_active ? '' : 'hidden'}">Alerta</span>
            </div>
          </div>

          <!-- Video Stream Container -->
          <div class="video-stream-container group relative">
            ${isEnabled ? `
              <img src="/api/cameras/${cam.id}/stream?fps=10" 
                   id="stream-img-${cam.id}"
                   class="video-stream-img" 
                   alt="${cam.name}" 
                   loading="lazy"
                   onerror="LiveGrid.handleStreamError('${cam.id}')" />
            ` : `
              <div class="w-full aspect-video bg-slate-950 flex flex-col items-center justify-center p-6 text-center space-y-3">
                <div class="w-12 h-12 rounded-full bg-slate-800/80 border border-slate-700 flex items-center justify-center text-slate-400">
                  <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                </div>
                <div>
                  <p class="text-xs font-semibold text-slate-200">Cámara Desactivada</p>
                  <p class="text-[11px] text-slate-500">Transmisión e inferencia IA temporalmente detenidas</p>
                </div>
                <button onclick="LiveGrid.toggleCamera('${cam.id}', true)" class="px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-md text-xs font-medium shadow transition flex items-center space-x-1.5">
                  <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                  <span>Reanudar Cámara</span>
                </button>
              </div>
            `}

            <!-- Floating Controls Overlay on Hover -->
            <div class="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center space-x-3 pointer-events-none group-hover:pointer-events-auto">
              <button onclick="LiveGrid.toggleCamera('${cam.id}', ${!isEnabled})" class="p-2 ${isEnabled ? 'bg-amber-900/80 hover:bg-amber-800 text-amber-200 border-amber-700' : 'bg-emerald-900/80 hover:bg-emerald-800 text-emerald-200 border-emerald-700'} rounded-full border shadow" title="${isEnabled ? 'Pausar / Desactivar Cámara' : 'Activar Cámara'}">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18.364 5.636a9 9 0 010 12.728m0 0l-2.829-2.829m2.829 2.829L12 12m-6.364 6.364a9 9 0 1112.728 0m0 0l-2.829-2.829M12 2v10" /></svg>
              </button>
              <button onclick="LiveGrid.downloadSnapshot('${cam.id}')" class="p-2 bg-slate-900/80 hover:bg-slate-800 text-slate-200 rounded-full border border-slate-700 shadow" title="Descargar Captura">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
              </button>
              <button onclick="LiveGrid.openRoiEditor('${cam.id}')" class="p-2 bg-slate-900/80 hover:bg-slate-800 text-slate-200 rounded-full border border-slate-700 shadow" title="Configurar Región de Interés">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" /></svg>
              </button>
              <button onclick="LiveGrid.editCamera('${cam.id}')" class="p-2 bg-blue-900/80 hover:bg-blue-800 text-blue-200 rounded-full border border-blue-700 shadow" title="Editar Cámara">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" /></svg>
              </button>
              <button onclick="LiveGrid.deleteCamera('${cam.id}', '${cam.name.replace(/'/g, "\\'")}')" class="p-2 bg-red-900/80 hover:bg-red-800 text-red-200 rounded-full border border-red-700 shadow" title="Eliminar Cámara">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
              </button>
              <button onclick="LiveGrid.toggleFullscreen('cam-card-${cam.id}')" class="p-2 bg-slate-900/80 hover:bg-slate-800 text-slate-200 rounded-full border border-slate-700 shadow" title="Pantalla Completa">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" /></svg>
              </button>
            </div>
          </div>
        </div>
      `;
    }).join('');
  },

  async updateTelemetry() {
    try {
      const res = await fetch('/api/cameras');
      if (!res.ok) return;
      const data = await res.json();
      this.cameras = data;

      data.forEach(cam => {
        const card = document.getElementById(`cam-card-${cam.id}`);
        if (!card) return;

        // Update card border pulse
        if (cam.alert_active) {
          card.classList.add('alert-active');
          card.classList.remove('motion-active');
        } else if (cam.motion_detected) {
          card.classList.add('motion-active');
          card.classList.remove('alert-active');
        } else {
          card.classList.remove('alert-active', 'motion-active');
        }

        // Update badges
        const fpsBadge = document.getElementById(`fps-badge-${cam.id}`);
        if (fpsBadge) {
          const fpsVal = (cam.current_fps || cam.effective_fps || cam.fps_target).toFixed(1);
          fpsBadge.textContent = `${fpsVal} FPS`;
        }

        const ecoBadge = document.getElementById(`eco-badge-${cam.id}`);
        if (ecoBadge) {
          if (cam.eco_mode) ecoBadge.classList.remove('hidden');
          else ecoBadge.classList.add('hidden');
        }

        const motionBadge = document.getElementById(`motion-badge-${cam.id}`);
        if (motionBadge) {
          if (cam.motion_detected) motionBadge.classList.remove('hidden');
          else motionBadge.classList.add('hidden');
        }

        const alertBadge = document.getElementById(`alert-badge-${cam.id}`);
        if (alertBadge) {
          if (cam.alert_active) alertBadge.classList.remove('hidden');
          else alertBadge.classList.add('hidden');
        }
      });
    } catch (e) {
      console.debug('Telemetry update error', e);
    }
  },

  handleStreamError(camId) {
    console.warn(`Stream error for camera ${camId}`);
  },

  downloadSnapshot(camId) {
    const a = document.createElement('a');
    a.href = `/api/cameras/${camId}/snapshot`;
    a.download = `snapshot_${camId}_${Date.now()}.jpg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    App.showToast('Descargando captura de imagen...', 'success');
  },

  openRoiEditor(camId) {
    App.switchTab('roi');
    if (window.RoiEditor) {
      RoiEditor.selectCamera(camId);
    }
  },

  toggleFullscreen(elementId) {
    const el = document.getElementById(elementId);
    if (!el) return;
    if (!document.fullscreenElement) {
      el.requestFullscreen().catch(err => console.error(err));
    } else {
      document.exitFullscreen().catch(err => console.error(err));
    }
  },

  editCamera(camId) {
    if (window.SettingsView) {
      App.switchTab('settings');
      SettingsView.openEditCameraModal(camId);
    }
  },

  deleteCamera(camId, camName) {
    const img = document.getElementById(`stream-img-${camId}`);
    if (img) img.src = '';
    if (window.SettingsView) {
      SettingsView.deleteCamera(camId, camName);
    }
  },

  async toggleCamera(camId, enable) {
    const img = document.getElementById(`stream-img-${camId}`);
    if (img && !enable) {
      img.src = '';
    }
    try {
      const res = await fetch(`/api/cameras/${camId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: Boolean(enable) }),
      });
      if (res.ok) {
        App.showToast(enable ? 'Cámara activada y transmitiendo' : 'Cámara pausada / desactivada', enable ? 'success' : 'info');
        this.fetchCameras();
        if (window.SettingsView) {
          SettingsView.loadCameraList();
        }
      } else {
        const err = await res.json();
        App.showToast(`Error: ${err.detail || 'No se pudo cambiar el estado de la cámara'}`, 'error');
      }
    } catch (e) {
      console.error(e);
      App.showToast('Error de red al actualizar cámara', 'error');
    }
  }
};

window.LiveGrid = LiveGrid;
