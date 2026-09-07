/**
 * Smart NVR — Events History Gallery, Filtering & HTML5 Video Modal Player
 */

const EventsGallery = {
  currentPage: 1,
  pageSize: 12,
  totalPages: 1,
  totalEvents: 0,
  activeEvent: null,

  init() {
    this.setupFilters();
    this.setupModal();
    this.fetchCamerasForFilter();
    this.fetchEvents();
  },

  onActivate() {
    this.fetchCamerasForFilter();
    this.fetchEvents();
  },

  setupFilters() {
    const slider = document.getElementById('filter-confidence');
    const label = document.getElementById('conf-val-label');
    if (slider && label) {
      slider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        label.textContent = val === 0 ? 'Todas (0%)' : `${Math.round(val * 100)}%`;
      });
    }

    const btnApply = document.getElementById('btn-apply-filters');
    if (btnApply) {
      btnApply.addEventListener('click', () => {
        this.currentPage = 1;
        this.fetchEvents();
      });
    }

    const btnReset = document.getElementById('btn-reset-filters');
    if (btnReset) {
      btnReset.addEventListener('click', () => {
        document.getElementById('filter-camera').value = '';
        document.getElementById('filter-class').value = '';
        document.getElementById('filter-confidence').value = '0.0';
        document.getElementById('conf-val-label').textContent = 'Todas (0%)';
        this.currentPage = 1;
        this.fetchEvents();
      });
    }

    const btnRefresh = document.getElementById('btn-refresh-events');
    if (btnRefresh) {
      btnRefresh.addEventListener('click', () => {
        this.fetchEvents();
        App.showToast('Actualizando historial de eventos...', 'info');
      });
    }

    const btnPurge = document.getElementById('btn-purge-orphans');
    if (btnPurge) {
      btnPurge.addEventListener('click', async () => {
        if (!confirm('¿Desea eliminar grabaciones huérfanas de cámaras borradas y archivos corruptos para liberar espacio?')) return;
        try {
          btnPurge.disabled = true;
          btnPurge.classList.add('opacity-50');
          App.showToast('Purgando archivos huérfanos...', 'info');
          const res = await fetch('/api/events/purge-orphaned', { method: 'POST' });
          const data = await res.json();
          if (res.ok) {
            App.showToast(data.message, 'success');
            if (App.pollHealth) App.pollHealth();
            this.fetchEvents();
          } else {
            App.showToast(data.detail || 'Error al purgar archivos', 'error');
          }
        } catch (e) {
          App.showToast('Error de red al purgar', 'error');
        } finally {
          btnPurge.disabled = false;
          btnPurge.classList.remove('opacity-50');
        }
      });
    }

    const btnSync = document.getElementById('btn-sync-disk');
    if (btnSync) {
      btnSync.addEventListener('click', async () => {
        try {
          btnSync.disabled = true;
          btnSync.classList.add('opacity-50');
          App.showToast('Buscando y sincronizando videos en disco...', 'info');
          const res = await fetch('/api/events/sync-disk', { method: 'POST' });
          const data = await res.json();
          if (res.ok) {
            App.showToast(data.message, 'success');
            this.fetchCamerasForFilter();
            this.fetchEvents();
          } else {
            App.showToast(data.detail || 'Error al sincronizar videos', 'error');
          }
        } catch (e) {
          App.showToast('Error de red al sincronizar', 'error');
        } finally {
          btnSync.disabled = false;
          btnSync.classList.remove('opacity-50');
        }
      });
    }

    // Pagination
    const btnPrev = document.getElementById('pag-prev');
    const btnNext = document.getElementById('pag-next');
    if (btnPrev) {
      btnPrev.addEventListener('click', () => {
        if (this.currentPage > 1) {
          this.currentPage--;
          this.fetchEvents();
        }
      });
    }
    if (btnNext) {
      btnNext.addEventListener('click', () => {
        if (this.currentPage < this.totalPages) {
          this.currentPage++;
          this.fetchEvents();
        }
      });
    }
  },

  setupModal() {
    const modal = document.getElementById('video-modal');
    const closeBtn = document.getElementById('modal-close-btn');
    const deleteBtn = document.getElementById('modal-delete-btn');

    if (closeBtn) {
      closeBtn.addEventListener('click', () => this.closeModal());
    }

    if (modal) {
      modal.addEventListener('click', (e) => {
        if (e.target === modal) this.closeModal();
      });
    }

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') this.closeModal();
    });

    if (deleteBtn) {
      deleteBtn.addEventListener('click', async () => {
        if (!this.activeEvent) return;
        await this.deleteEvent(this.activeEvent.id);
      });
    }
  },

  async fetchCamerasForFilter() {
    try {
      const res = await fetch('/api/cameras');
      if (!res.ok) return;
      const cameras = await res.json();
      const select = document.getElementById('filter-camera');
      if (!select) return;

      const currentVal = select.value;
      select.innerHTML = '<option value="">Todas las cámaras</option>' +
        cameras.map(c => `<option value="${c.id}">${c.name}</option>`).join('');
      select.value = currentVal;
    } catch (e) {
      console.error(e);
    }
  },

  async fetchEvents() {
    const cam = document.getElementById('filter-camera')?.value;
    const cls = document.getElementById('filter-class')?.value;
    const conf = document.getElementById('filter-confidence')?.value;

    const params = new URLSearchParams({
      page: this.currentPage,
      page_size: this.pageSize,
    });
    if (cam) params.append('camera_id', cam);
    if (cls) params.append('class_name', cls);
    if (conf && parseFloat(conf) > 0) params.append('min_confidence', conf);

    try {
      const res = await fetch(`/api/events?${params.toString()}`);
      if (!res.ok) return;
      const data = await res.json();

      this.totalEvents = data.total;
      this.totalPages = data.total_pages;
      this.currentPage = data.page;

      this.renderEvents(data.items);
      this.updatePaginationUI();
    } catch (e) {
      console.error('Failed to fetch events', e);
    }
  },

  renderEvents(events) {
    const container = document.getElementById('events-cards-container');
    if (!container) return;

    if (!events || events.length === 0) {
      container.innerHTML = `
        <div class="col-span-full py-16 text-center text-slate-500 text-xs">
          No se encontraron incidentes con los filtros seleccionados.
        </div>
      `;
      return;
    }

    container.innerHTML = events.map(evt => {
      const cls = evt.detection_class || 'person';
      const isPerson = cls.toLowerCase() === 'person';
      const badgeClass = isPerson ? 'badge-person' : 'badge-vehicle';
      const label = isPerson ? 'Persona' : (cls === 'car' ? 'Automóvil' : cls);
      const confPct = Math.round(evt.max_confidence * 100);
      const snapUrl = `/api/events/${evt.id}/snapshot`;

      return `
        <div class="bg-slate-900 border border-slate-800 hover:border-slate-700 rounded-lg overflow-hidden shadow-lg transition-all duration-200 flex flex-col group cursor-pointer"
             onclick="EventsGallery.openEventModal('${evt.id}')">
          <!-- Thumbnail Container with Play Overlay -->
          <div class="relative aspect-video bg-black overflow-hidden">
            <img src="${snapUrl}" alt="${evt.id}" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" loading="lazy" />
            <div class="absolute inset-0 bg-black/30 group-hover:bg-black/10 transition-colors flex items-center justify-center">
              <div class="w-10 h-10 rounded-full bg-blue-600/90 group-hover:bg-blue-600 text-white flex items-center justify-center shadow-lg transform group-hover:scale-110 transition-transform">
                <svg class="w-5 h-5 translate-x-0.5" fill="currentColor" viewBox="0 0 20 20"><path d="M6.3 2.841A1.5 1.5 0 004 4.11v11.78a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" /></svg>
              </div>
            </div>
            <!-- Duration Badge -->
            <span class="absolute bottom-2 right-2 px-1.5 py-0.5 bg-black/70 text-white font-mono text-[10px] rounded">
              ${evt.duration_seconds ? evt.duration_seconds.toFixed(1) : '0'}s
            </span>
          </div>

          <!-- Card Content -->
          <div class="p-3.5 flex-1 flex flex-col justify-between text-xs space-y-2">
            <div>
              <div class="flex items-center justify-between mb-1">
                <span class="status-pill ${badgeClass}">${label} ${confPct}%</span>
                <span class="text-slate-400 text-[11px]">${evt.camera_name || evt.camera_id}</span>
              </div>
              <p class="text-slate-200 font-semibold text-xs truncate" title="${evt.id}">
                ${evt.start_time}
              </p>
            </div>
            <div class="flex items-center justify-between text-[11px] text-slate-500 pt-1 border-t border-slate-800/80">
              <span>${(evt.file_size_bytes / (1024 * 1024)).toFixed(1)} MB</span>
              <div class="flex items-center space-x-2">
                <button onclick="event.stopPropagation(); EventsGallery.deleteEvent('${evt.id}')" 
                        class="p-1 text-slate-500 hover:text-red-400 rounded hover:bg-slate-800 transition" 
                        title="Eliminar este clip y registro">
                  <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                </button>
                <span class="text-blue-400 font-medium group-hover:underline">Ver Clip &rarr;</span>
              </div>
            </div>
          </div>
        </div>
      `;
    }).join('');
  },

  updatePaginationUI() {
    const elTotal = document.getElementById('pag-total');
    const elStart = document.getElementById('pag-start');
    const elEnd = document.getElementById('pag-end');
    const elInfo = document.getElementById('pag-page-info');
    const btnPrev = document.getElementById('pag-prev');
    const btnNext = document.getElementById('pag-next');

    const start = this.totalEvents === 0 ? 0 : (this.currentPage - 1) * this.pageSize + 1;
    const end = Math.min(this.currentPage * this.pageSize, this.totalEvents);

    if (elTotal) elTotal.textContent = this.totalEvents;
    if (elStart) elStart.textContent = start;
    if (elEnd) elEnd.textContent = end;
    if (elInfo) elInfo.textContent = `Pág ${this.currentPage} / ${this.totalPages}`;

    if (btnPrev) btnPrev.disabled = this.currentPage <= 1;
    if (btnNext) btnNext.disabled = this.currentPage >= this.totalPages;
  },

  async openEventModal(eventId) {
    try {
      const res = await fetch(`/api/events/${eventId}`);
      if (!res.ok) return;
      const event = await res.json();
      this.activeEvent = event;

      const modal = document.getElementById('video-modal');
      const title = document.getElementById('modal-event-title');
      const subtitle = document.getElementById('modal-event-subtitle');
      const videoPlayer = document.getElementById('modal-video-player');
      const badgeClass = document.getElementById('modal-badge-class');
      const badgeDuration = document.getElementById('modal-badge-duration');
      const badgeSize = document.getElementById('modal-badge-size');
      const dlClip = document.getElementById('modal-download-clip');
      const dlSnap = document.getElementById('modal-download-snap');

      if (title) title.textContent = `Incidente ${event.id}`;
      if (subtitle) subtitle.textContent = `${event.camera_name || event.camera_id} — ${event.start_time}`;

      const cls = event.detection_class || 'person';
      if (badgeClass) {
        badgeClass.textContent = `${cls.toUpperCase()} (${Math.round(event.max_confidence * 100)}%)`;
        badgeClass.className = `status-pill ${cls.toLowerCase() === 'person' ? 'badge-person' : 'badge-vehicle'}`;
      }
      if (badgeDuration) badgeDuration.textContent = `${event.duration_seconds.toFixed(1)}s`;
      if (badgeSize) badgeSize.textContent = `${(event.file_size_bytes / (1024 * 1024)).toFixed(2)} MB`;

      const videoUrl = `/api/events/${event.id}/video`;
      const snapUrl = `/api/events/${event.id}/snapshot`;

      if (videoPlayer) {
        videoPlayer.src = videoUrl;
        videoPlayer.load();
        videoPlayer.play().catch(() => {});
      }

      if (dlClip) {
        dlClip.href = videoUrl;
        dlClip.download = `${event.id}.mp4`;
      }
      if (dlSnap) {
        dlSnap.href = snapUrl;
        dlSnap.download = `${event.id}_snapshot.jpg`;
      }

      if (modal) modal.classList.remove('hidden');
    } catch (e) {
      console.error('Failed to open event modal', e);
    }
  },

  async deleteEvent(eventId) {
    if (!confirm(`¿Está seguro de que desea eliminar el incidente "${eventId}" y borrar su archivo de video del disco?`)) {
      return;
    }
    if (this.activeEvent && this.activeEvent.id === eventId) {
      this.closeModal();
    }
    try {
      const res = await fetch(`/api/events/${eventId}`, { method: 'DELETE' });
      if (res.ok) {
        App.showToast('Clip de video y evento eliminados correctamente', 'success');
        this.fetchEvents();
      } else {
        const err = await res.json();
        App.showToast(`Error al eliminar: ${err.detail || 'Fallo desconocido'}`, 'error');
      }
    } catch (e) {
      console.error(e);
      App.showToast('Error de red al eliminar el evento', 'error');
    }
  },

  closeModal() {
    const modal = document.getElementById('video-modal');
    const videoPlayer = document.getElementById('modal-video-player');
    if (videoPlayer) {
      videoPlayer.pause();
      videoPlayer.src = '';
    }
    if (modal) modal.classList.add('hidden');
    this.activeEvent = null;
  }
};

window.EventsGallery = EventsGallery;
