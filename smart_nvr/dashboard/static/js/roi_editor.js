/**
 * Smart NVR — HTML5 Canvas Interactive ROI Polygon Editor & Threshold Controls
 */

const RoiEditor = {
  currentCameraId: null,
  currentPoints: [], // Array of [x, y] in [0.0, 1.0]
  image: new Image(),
  canvas: null,
  ctx: null,

  init() {
    this.canvas = document.getElementById('roi-canvas');
    if (this.canvas) {
      this.ctx = this.canvas.getContext('2d');
      this.canvas.addEventListener('click', (e) => this.handleCanvasClick(e));
    }

    this.setupControls();
  },

  onActivate() {
    this.populateCameraSelect();
  },

  selectCamera(camId) {
    this.currentCameraId = camId;
    const sel = document.getElementById('roi-camera-select');
    if (sel) sel.value = camId;
    this.loadCameraConfig(camId);
  },

  async populateCameraSelect() {
    try {
      const res = await fetch('/api/cameras');
      if (!res.ok) return;
      const cameras = await res.json();
      const select = document.getElementById('roi-camera-select');
      if (!select) return;

      select.innerHTML = cameras.map(c => `<option value="${c.id}">${c.name}</option>`).join('');

      if (cameras.length > 0) {
        if (!this.currentCameraId || !cameras.some(c => c.id === this.currentCameraId)) {
          this.currentCameraId = cameras[0].id;
        }
        select.value = this.currentCameraId;
        this.loadCameraConfig(this.currentCameraId);
      }
    } catch (e) {
      console.error(e);
    }
  },

  setupControls() {
    const sel = document.getElementById('roi-camera-select');
    if (sel) {
      sel.addEventListener('change', (e) => {
        this.currentCameraId = e.target.value;
        this.loadCameraConfig(this.currentCameraId);
      });
    }

    // Sliders
    const sliderSens = document.getElementById('slider-sensitivity');
    const labelSens = document.getElementById('label-sensitivity');
    if (sliderSens && labelSens) {
      sliderSens.addEventListener('input', (e) => {
        labelSens.textContent = `${Math.round(e.target.value * 100)}%`;
      });
    }

    const sliderMin = document.getElementById('slider-min-contour');
    const labelMin = document.getElementById('label-min-contour');
    if (sliderMin && labelMin) {
      sliderMin.addEventListener('input', (e) => {
        labelMin.textContent = `${e.target.value} px`;
      });
    }

    const sliderAi = document.getElementById('slider-ai-conf');
    const labelAi = document.getElementById('label-ai-conf');
    if (sliderAi && labelAi) {
      sliderAi.addEventListener('input', (e) => {
        labelAi.textContent = `${Math.round(e.target.value * 100)}%`;
      });
    }

    // Buttons
    const btnClear = document.getElementById('btn-roi-clear');
    if (btnClear) {
      btnClear.addEventListener('click', () => {
        this.currentPoints = [];
        this.redraw();
        App.showToast('Polígono ROI limpiado', 'info');
      });
    }

    const btnReload = document.getElementById('btn-roi-reload-frame');
    if (btnReload) {
      btnReload.addEventListener('click', () => {
        if (this.currentCameraId) {
          this.loadSnapshot(this.currentCameraId);
          App.showToast('Fotograma de referencia actualizado', 'info');
        }
      });
    }

    const btnSave = document.getElementById('btn-roi-save');
    if (btnSave) {
      btnSave.addEventListener('click', () => this.saveConfig());
    }
  },

  async loadCameraConfig(camId) {
    if (!camId) return;
    this.loadSnapshot(camId);

    try {
      const res = await fetch(`/api/cameras/${camId}/detection-config`);
      if (!res.ok) return;
      const data = await res.json();

      // Sliders & Checkboxes
      const sliderSens = document.getElementById('slider-sensitivity');
      const labelSens = document.getElementById('label-sensitivity');
      if (sliderSens) {
        sliderSens.value = data.motion_sensitivity !== undefined ? data.motion_sensitivity : 0.5;
        if (labelSens) labelSens.textContent = `${Math.round(sliderSens.value * 100)}%`;
      }

      const sliderMin = document.getElementById('slider-min-contour');
      const labelMin = document.getElementById('label-min-contour');
      if (sliderMin) {
        sliderMin.value = data.min_contour_area || 500;
        if (labelMin) labelMin.textContent = `${sliderMin.value} px`;
      }

      const sliderAi = document.getElementById('slider-ai-conf');
      const labelAi = document.getElementById('label-ai-conf');
      if (sliderAi) {
        sliderAi.value = data.confidence_threshold || 0.5;
        if (labelAi) labelAi.textContent = `${Math.round(sliderAi.value * 100)}%`;
      }

      const chkShadows = document.getElementById('chk-detect-shadows');
      if (chkShadows) chkShadows.checked = !!data.mog2_detect_shadows;

      const chkAi = document.getElementById('chk-ai-enabled');
      if (chkAi) chkAi.checked = !!data.ai_enabled;

      // Extract existing ROI polygon
      if (data.rois && data.rois.length > 0 && Array.isArray(data.rois[0])) {
        this.currentPoints = JSON.parse(JSON.stringify(data.rois[0]));
      } else {
        this.currentPoints = [];
      }

      this.redraw();
    } catch (e) {
      console.error('Failed to load detection config', e);
    }
  },

  loadSnapshot(camId) {
    const overlay = document.getElementById('roi-loading-overlay');
    if (overlay) overlay.classList.remove('hidden');

    this.image = new Image();
    this.image.crossOrigin = 'anonymous';
    this.image.src = `/api/cameras/${camId}/snapshot?t=${Date.now()}`;
    this.image.onload = () => {
      if (overlay) overlay.classList.add('hidden');
      this.resizeCanvas();
      this.redraw();
    };
    this.image.onerror = () => {
      if (overlay) overlay.textContent = 'No se pudo cargar la captura de la cámara.';
    };
  },

  resizeCanvas() {
    if (!this.canvas) return;
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width = rect.width;
    this.canvas.height = rect.height;
  },

  handleCanvasClick(e) {
    if (!this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;

    // Clamp inside [0, 1]
    const cx = Math.max(0, Math.min(1, x));
    const cy = Math.max(0, Math.min(1, y));

    this.currentPoints.push([parseFloat(cx.toFixed(4)), parseFloat(cy.toFixed(4))]);
    this.redraw();
  },

  redraw() {
    if (!this.ctx || !this.canvas) return;
    const w = this.canvas.width;
    const h = this.canvas.height;

    this.ctx.clearRect(0, 0, w, h);

    // Draw camera snapshot background
    if (this.image.complete && this.image.naturalWidth > 0) {
      this.ctx.drawImage(this.image, 0, 0, w, h);
    } else {
      this.ctx.fillStyle = '#0f172a';
      this.ctx.fillRect(0, 0, w, h);
    }

    if (this.currentPoints.length === 0) return;

    // Draw polygon lines & fill
    this.ctx.beginPath();
    this.currentPoints.forEach((pt, idx) => {
      const px = pt[0] * w;
      const py = pt[1] * h;
      if (idx === 0) this.ctx.moveTo(px, py);
      else this.ctx.lineTo(px, py);
    });

    if (this.currentPoints.length >= 3) {
      this.ctx.closePath();
      this.ctx.fillStyle = 'rgba(59, 130, 246, 0.25)'; // Blue semi-transparent fill
      this.ctx.fill();
    }

    this.ctx.strokeStyle = '#3b82f6';
    this.ctx.lineWidth = 2.5;
    this.ctx.stroke();

    // Draw vertex dots
    this.currentPoints.forEach((pt, idx) => {
      const px = pt[0] * w;
      const py = pt[1] * h;
      this.ctx.beginPath();
      this.ctx.arc(px, py, 5, 0, 2 * Math.PI);
      this.ctx.fillStyle = idx === 0 ? '#10b981' : '#f59e0b';
      this.ctx.fill();
      this.ctx.strokeStyle = '#ffffff';
      this.ctx.lineWidth = 1.5;
      this.ctx.stroke();
    });
  },

  async saveConfig() {
    if (!this.currentCameraId) return;

    const sliderSens = document.getElementById('slider-sensitivity');
    const sliderMin = document.getElementById('slider-min-contour');
    const sliderAi = document.getElementById('slider-ai-conf');
    const chkShadows = document.getElementById('chk-detect-shadows');
    const chkAi = document.getElementById('chk-ai-enabled');

    const rois = this.currentPoints.length >= 3 ? [this.currentPoints] : [];

    const payload = {
      rois: rois,
      motion_sensitivity: sliderSens ? parseFloat(sliderSens.value) : 0.5,
      min_contour_area: sliderMin ? parseInt(sliderMin.value) : 500,
      confidence_threshold: sliderAi ? parseFloat(sliderAi.value) : 0.5,
      mog2_detect_shadows: chkShadows ? chkShadows.checked : true,
      ai_enabled: chkAi ? chkAi.checked : true,
    };

    try {
      const res = await fetch(`/api/cameras/${this.currentCameraId}/detection-config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        App.showToast('Configuración y ROI guardados exitosamente', 'success');
      } else {
        App.showToast('Error al guardar configuración', 'error');
      }
    } catch (e) {
      App.showToast('Error de red al guardar', 'error');
    }
  }
};

window.RoiEditor = RoiEditor;
window.addEventListener('resize', () => {
  if (RoiEditor.canvas) {
    RoiEditor.resizeCanvas();
    RoiEditor.redraw();
  }
});
