/**
 * Smart NVR — Settings View, Gmail SMTP Alerting & Camera Registration
 */

const SettingsView = {
  camerasCache: [],

  init() {
    this.setupSmtpForm();
    this.setupCameraForm();
    this.setupEditModal();
    this.setupCameraListRefresh();
  },

  onActivate() {
    this.fetchSettings();
    this.loadCameraList();
  },

  setupSmtpForm() {
    const sliderCd = document.getElementById('slider-cooldown');
    const labelCd = document.getElementById('label-cooldown');
    if (sliderCd && labelCd) {
      sliderCd.addEventListener('input', (e) => {
        labelCd.textContent = `${e.target.value} segundos`;
      });
    }

    const form = document.getElementById('form-settings-smtp');
    if (form) {
      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await this.saveSettings();
      });
    }

    const btnTest = document.getElementById('btn-test-email');
    if (btnTest) {
      btnTest.addEventListener('click', async () => {
        await this.sendTestEmail();
      });
    }
  },

  setupCameraForm() {
    const btnProbe = document.getElementById('btn-probe-camera');
    if (btnProbe) {
      btnProbe.addEventListener('click', async () => {
        await this.probeCamera();
      });
    }

    const selectType = document.getElementById('cam-type');
    const inputUrl = document.getElementById('cam-url');
    if (selectType && inputUrl) {
      selectType.addEventListener('change', (e) => {
        const val = e.target.value;
        if (val === 'usb') {
          inputUrl.value = '0';
          inputUrl.placeholder = '0 (Cámara integrada o USB principal)';
        } else if (val === 'synthetic') {
          inputUrl.value = 'synthetic://moving_person';
          inputUrl.placeholder = 'synthetic://moving_person';
        } else if (val === 'rtsp') {
          inputUrl.value = '';
          inputUrl.placeholder = 'rtsp://user:pass@192.168.1.100:554/stream';
        } else if (val === 'file') {
          inputUrl.value = '';
          inputUrl.placeholder = 'storage/video.mp4 o C:\\ruta\\video.mp4';
        }
      });
    }

    const form = document.getElementById('form-add-camera');
    if (form) {
      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await this.registerCamera();
      });
    }
  },

  async fetchSettings() {
    try {
      const res = await fetch('/api/settings');
      if (!res.ok) return;
      const data = await res.json();

      document.getElementById('smtp-server').value = data.smtp_server || 'smtp.gmail.com';
      document.getElementById('smtp-port').value = data.smtp_port || 587;
      document.getElementById('smtp-user').value = data.smtp_username || '';
      document.getElementById('smtp-pass').value = data.smtp_password || '';
      document.getElementById('alert-recipients').value = (data.alert_recipients || []).join(', ');

      const sliderCd = document.getElementById('slider-cooldown');
      const labelCd = document.getElementById('label-cooldown');
      if (sliderCd) {
        sliderCd.value = data.alert_cooldown_seconds || 60;
        if (labelCd) labelCd.textContent = `${sliderCd.value} segundos`;
      }
    } catch (e) {
      console.error('Failed to load settings', e);
    }
  },

  async saveSettings() {
    const server = document.getElementById('smtp-server').value.trim();
    const port = parseInt(document.getElementById('smtp-port').value);
    const user = document.getElementById('smtp-user').value.trim();
    const pass = document.getElementById('smtp-pass').value.trim();
    const recipientsRaw = document.getElementById('alert-recipients').value;
    const recipients = recipientsRaw.split(',').map(r => r.trim()).filter(r => r.length > 0);
    const cooldown = parseInt(document.getElementById('slider-cooldown').value);

    const payload = {
      smtp_server: server,
      smtp_port: port,
      smtp_username: user,
      smtp_from_email: user,
      alert_recipients: recipients,
      alert_cooldown_seconds: cooldown,
    };
    if (pass && pass !== '********') {
      payload.smtp_password = pass;
    }

    try {
      const res = await fetch('/api/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (res.ok) {
        App.showToast('Configuración SMTP guardada exitosamente', 'success');
        this.fetchSettings();
      } else {
        App.showToast('Error al guardar configuración', 'error');
      }
    } catch (e) {
      App.showToast('Error de red al guardar', 'error');
    }
  },

  async sendTestEmail() {
    const banner = document.getElementById('email-test-result');
    if (banner) {
      banner.className = 'p-3 rounded text-xs bg-slate-800 border border-slate-700 text-slate-300';
      banner.textContent = 'Enviando correo de prueba a través de SMTP...';
      banner.classList.remove('hidden');
    }

    try {
      const res = await fetch('/api/settings/test-email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      const data = await res.json();

      if (banner) {
        if (data.success) {
          banner.className = 'p-3 rounded text-xs bg-emerald-950/80 border border-emerald-800 text-emerald-300';
          banner.innerHTML = `<strong>Éxito:</strong> ${data.message}`;
          App.showToast('Correo de prueba enviado con éxito', 'success');
        } else {
          banner.className = 'p-3 rounded text-xs bg-red-950/80 border border-red-800 text-red-300';
          banner.innerHTML = `<strong>Error:</strong> ${data.message}`;
          App.showToast('Error en la prueba de correo', 'error');
        }
      }
    } catch (e) {
      if (banner) {
        banner.className = 'p-3 rounded text-xs bg-red-950/80 border border-red-800 text-red-300';
        banner.textContent = 'Error de red al intentar despachar correo de prueba.';
      }
      App.showToast('Error de conexión con el servidor', 'error');
    }
  },

  async probeCamera() {
    const type = document.getElementById('cam-type').value;
    const url = document.getElementById('cam-url').value.trim();
    const fps = parseInt(document.getElementById('cam-fps').value) || 15;
    const box = document.getElementById('probe-result-box');

    if (box) {
      box.className = 'p-2.5 rounded bg-slate-800 border border-slate-700 text-slate-300 text-[11px]';
      box.textContent = 'Probando conexión con la fuente de video...';
      box.classList.remove('hidden');
    }

    try {
      const res = await fetch('/api/cameras/test-connection', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_type: type,
          source_url: url,
          fps_target: fps,
        }),
      });
      const data = await res.json();

      if (box) {
        if (data.valid) {
          box.className = 'p-2.5 rounded bg-emerald-950/70 border border-emerald-800 text-emerald-300 text-[11px]';
          box.innerHTML = `
            <strong>Conexión Exitosa:</strong> Resolución: ${data.width}x${data.height}, Tasa: ${data.fps} FPS.
            <div class="mt-1 text-slate-400">${data.message}</div>
          `;
          App.showToast('Fuente de video validada correctamente', 'success');
        } else {
          box.className = 'p-2.5 rounded bg-red-950/70 border border-red-800 text-red-300 text-[11px]';
          box.innerHTML = `<strong>Fallo de Conexión:</strong> ${data.message}`;
          App.showToast('No se pudo verificar la fuente', 'error');
        }
      }
    } catch (e) {
      if (box) {
        box.className = 'p-2.5 rounded bg-red-950/70 border border-red-800 text-red-300 text-[11px]';
        box.textContent = 'Error de red al consultar el probador de cámaras.';
      }
    }
  },

  async registerCamera() {
    const name = document.getElementById('cam-name').value.trim();
    const type = document.getElementById('cam-type').value;
    const url = document.getElementById('cam-url').value.trim();
    const fps = parseInt(document.getElementById('cam-fps').value) || 15;

    if (!name || !url) {
      App.showToast('Por favor ingrese nombre y URL de fuente', 'warning');
      return;
    }

    const payload = {
      name: name,
      source_type: type,
      source_url: url,
      enabled: true,
      fps_target: fps,
      rois: [],
    };

    try {
      const res = await fetch('/api/cameras', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        App.showToast(`Cámara '${name}' registrada y activada`, 'success');
        document.getElementById('form-add-camera').reset();
        document.getElementById('cam-url').value = 'synthetic://moving_person';
        const box = document.getElementById('probe-result-box');
        if (box) box.classList.add('hidden');

        // Refresh list and switch to live grid
        this.loadCameraList();
        App.switchTab('live');
        if (window.LiveGrid) LiveGrid.fetchCameras();
        if (window.RoiEditor) RoiEditor.loadCameras();
      } else {
        const err = await res.json();
        App.showToast(`Error al registrar cámara: ${err.detail || 'Fallo desconocido'}`, 'error');
      }
    } catch (e) {
      App.showToast('Error de red al registrar cámara', 'error');
    }
  },

  setupCameraListRefresh() {
    const btn = document.getElementById('btn-refresh-cam-list');
    if (btn) {
      btn.addEventListener('click', () => {
        this.loadCameraList();
        App.showToast('Lista de cámaras actualizada', 'info');
      });
    }
  },

  async loadCameraList() {
    const container = document.getElementById('settings-cameras-container');
    if (!container) return;

    try {
      const res = await fetch('/api/cameras');
      if (!res.ok) {
        container.innerHTML = '<div class="text-center py-6 text-red-400 text-xs">Error al cargar la lista de cámaras.</div>';
        return;
      }
      const cameras = await res.json();
      this.camerasCache = cameras;

      if (cameras.length === 0) {
        container.innerHTML = `
          <div class="text-center py-8 text-slate-500 text-xs bg-slate-950/40 rounded-lg border border-slate-800/80 p-4">
            No hay cámaras registradas en el sistema. Registre una nueva cámara arriba.
          </div>
        `;
        return;
      }

      const sourceTypeLabels = {
        file: { text: 'Archivo de Video', badge: 'bg-purple-900/40 text-purple-300 border border-purple-800/50' },
        rtsp: { text: 'Cámara IP (RTSP)', badge: 'bg-blue-900/40 text-blue-300 border border-blue-800/50' },
        usb: { text: 'Cámara Web USB', badge: 'bg-emerald-900/40 text-emerald-300 border border-emerald-800/50' },
        synthetic: { text: 'Simulador', badge: 'bg-amber-900/40 text-amber-300 border border-amber-800/50' },
      };

      container.innerHTML = `
        <div class="overflow-x-auto">
          <table class="w-full text-left border-collapse text-xs">
            <thead>
              <tr class="border-b border-slate-800 text-slate-400 font-semibold text-[11px]">
                <th class="py-2.5 px-3">Cámara / ID</th>
                <th class="py-2.5 px-3">Tipo de Fuente</th>
                <th class="py-2.5 px-3">Ruta o URL</th>
                <th class="py-2.5 px-3 text-center">FPS</th>
                <th class="py-2.5 px-3 text-center">Estado</th>
                <th class="py-2.5 px-3 text-right">Acciones</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-800/60">
              ${cameras.map(cam => {
                const typeInfo = sourceTypeLabels[cam.source_type] || { text: cam.source_type, badge: 'bg-slate-800 text-slate-300' };
                const isRunning = Boolean(cam.is_running);
                const isEnabled = Boolean(cam.enabled);

                return `
                  <tr class="hover:bg-slate-800/30 transition-colors">
                    <td class="py-3 px-3">
                      <div class="font-bold text-slate-200">${cam.name}</div>
                      <div class="text-[10px] text-slate-500 font-mono">${cam.id}</div>
                    </td>
                    <td class="py-3 px-3">
                      <span class="px-2 py-0.5 rounded text-[10px] font-medium ${typeInfo.badge}">${typeInfo.text}</span>
                    </td>
                    <td class="py-3 px-3">
                      <div class="max-w-[220px] sm:max-w-xs truncate text-slate-300 font-mono text-[11px]" title="${cam.source_url}">${cam.source_url}</div>
                    </td>
                    <td class="py-3 px-3 text-center font-mono text-slate-300">
                      ${cam.fps_target || 15}
                    </td>
                    <td class="py-3 px-3 text-center">
                      <span class="inline-flex items-center space-x-1 px-2 py-0.5 rounded-full text-[10px] font-semibold ${isRunning ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-slate-800 text-slate-400 border border-slate-700'}">
                        <span class="w-1.5 h-1.5 rounded-full ${isRunning ? 'bg-emerald-400 animate-pulse' : 'bg-slate-500'}"></span>
                        <span>${isRunning ? 'ONLINE' : (isEnabled ? 'DETENIDA' : 'DESHABILITADA')}</span>
                      </span>
                    </td>
                    <td class="py-3 px-3 text-right space-x-1 whitespace-nowrap">
                      <!-- Toggle Enabled -->
                      <button onclick="SettingsView.toggleCameraEnabled('${cam.id}', ${isEnabled})" 
                              class="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 transition text-[11px]"
                              title="${isEnabled ? 'Pausar transmisión de esta cámara' : 'Activar transmisión de esta cámara'}">
                        ${isEnabled ? '⏸ Pausar' : '▶ Activar'}
                      </button>
                      <!-- Edit Camera -->
                      <button onclick="SettingsView.openEditCameraModal('${cam.id}')" 
                              class="px-2.5 py-1 rounded bg-blue-900/40 hover:bg-blue-900/70 border border-blue-700/60 text-blue-300 transition text-[11px]"
                              title="Editar parámetros de la cámara">
                        ✏ Editar
                      </button>
                      <!-- Delete Camera -->
                      <button onclick="SettingsView.deleteCamera('${cam.id}', '${cam.name.replace(/'/g, "\\'")}')" 
                              class="px-2.5 py-1 rounded bg-red-900/40 hover:bg-red-900/70 border border-red-800/60 text-red-300 transition text-[11px]"
                              title="Eliminar esta cámara permanentemente">
                        🗑 Eliminar
                      </button>
                    </td>
                  </tr>
                `;
              }).join('')}
            </tbody>
          </table>
        </div>
      `;
    } catch (e) {
      console.error('Error loading cameras list', e);
      container.innerHTML = '<div class="text-center py-6 text-red-400 text-xs">Error de conexión al cargar la lista de cámaras.</div>';
    }
  },

  setupEditModal() {
    const modal = document.getElementById('edit-camera-modal');
    const btnClose = document.getElementById('btn-close-edit-modal');
    const btnCancel = document.getElementById('btn-cancel-edit-cam');
    const form = document.getElementById('form-edit-camera');

    const closeModal = () => {
      if (modal) modal.classList.add('hidden');
    };

    if (btnClose) btnClose.addEventListener('click', closeModal);
    if (btnCancel) btnCancel.addEventListener('click', closeModal);

    if (form) {
      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await this.saveEditCamera();
      });
    }
  },

  async openEditCameraModal(camId) {
    let cam = this.camerasCache.find(c => c.id === camId);
    if (!cam) {
      try {
        const res = await fetch(`/api/cameras/${camId}`);
        if (res.ok) cam = await res.json();
      } catch (e) {
        console.error('Failed to fetch camera details', e);
      }
    }

    if (!cam) {
      App.showToast('No se encontraron los datos de la cámara', 'error');
      return;
    }

    document.getElementById('edit-cam-id').value = cam.id;
    document.getElementById('edit-cam-id-display').textContent = `ID: ${cam.id}`;
    document.getElementById('edit-cam-name').value = cam.name || '';
    document.getElementById('edit-cam-type').value = cam.source_type || 'synthetic';
    document.getElementById('edit-cam-fps').value = cam.fps_target || 15;
    document.getElementById('edit-cam-url').value = cam.source_url || '';
    document.getElementById('edit-cam-enabled').checked = Boolean(cam.enabled !== false);

    const modal = document.getElementById('edit-camera-modal');
    if (modal) modal.classList.remove('hidden');
  },

  async saveEditCamera() {
    const camId = document.getElementById('edit-cam-id').value;
    const name = document.getElementById('edit-cam-name').value.trim();
    const type = document.getElementById('edit-cam-type').value;
    const fps = parseInt(document.getElementById('edit-cam-fps').value) || 15;
    const url = document.getElementById('edit-cam-url').value.trim();
    const enabled = document.getElementById('edit-cam-enabled').checked;

    if (!name || !url) {
      App.showToast('El nombre y la URL de la fuente son obligatorios', 'warning');
      return;
    }

    const payload = {
      name: name,
      source_type: type,
      source_url: url,
      fps_target: fps,
      enabled: enabled,
    };

    try {
      const res = await fetch(`/api/cameras/${camId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        App.showToast(`Cámara '${name}' actualizada exitosamente`, 'success');
        const modal = document.getElementById('edit-camera-modal');
        if (modal) modal.classList.add('hidden');

        // Refresh views
        this.loadCameraList();
        if (window.LiveGrid) LiveGrid.fetchCameras();
        if (window.RoiEditor) RoiEditor.loadCameras();
      } else {
        const err = await res.json();
        App.showToast(`Error al actualizar cámara: ${err.detail || 'Fallo desconocido'}`, 'error');
      }
    } catch (e) {
      App.showToast('Error de red al actualizar la cámara', 'error');
    }
  },

  async deleteCamera(camId, camName) {
    const confirmed = confirm(
      `¿Estás seguro de que deseas eliminar permanentemente la cámara "${camName}" (${camId})?\n\n` +
      `Se detendrá la transmisión en vivo y se eliminará su configuración.`
    );
    if (!confirmed) return;

    const streamImg = document.getElementById(`stream-img-${camId}`);
    if (streamImg) streamImg.src = '';

    try {
      const res = await fetch(`/api/cameras/${camId}`, {
        method: 'DELETE',
      });

      if (res.ok) {
        App.showToast(`Cámara "${camName}" eliminada correctamente`, 'success');
        this.loadCameraList();
        if (window.LiveGrid) LiveGrid.fetchCameras();
        if (window.RoiEditor) RoiEditor.loadCameras();
      } else {
        const err = await res.json();
        App.showToast(`Error al eliminar cámara: ${err.detail || 'Fallo desconocido'}`, 'error');
      }
    } catch (e) {
      App.showToast('Error de red al eliminar la cámara', 'error');
    }
  },

  async toggleCameraEnabled(camId, currentEnabled) {
    if (currentEnabled) {
      const streamImg = document.getElementById(`stream-img-${camId}`);
      if (streamImg) streamImg.src = '';
    }
    try {
      const res = await fetch(`/api/cameras/${camId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: !currentEnabled }),
      });

      if (res.ok) {
        App.showToast(currentEnabled ? 'Cámara pausada' : 'Cámara reanudada y activa', 'info');
        this.loadCameraList();
        if (window.LiveGrid) LiveGrid.fetchCameras();
      } else {
        App.showToast('Error al cambiar estado de la cámara', 'error');
      }
    } catch (e) {
      App.showToast('Error de red al cambiar estado de la cámara', 'error');
    }
  }
};

window.SettingsView = SettingsView;
