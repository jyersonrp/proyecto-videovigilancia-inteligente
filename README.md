# Smart NVR — Sistema de Videovigilancia Inteligente Modular

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue?logo=python" alt="Python Version" />
  <img src="https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/OpenCV-4.10%2B-5C3EE8?logo=opencv" alt="OpenCV" />
  <img src="https://img.shields.io/badge/YOLOv8-ONNX%20Runtime%20CPU-00C4CC" alt="YOLOv8 ONNX" />
  <img src="https://img.shields.io/badge/License-MIT-brightgreen" alt="MIT License" />
</p>

Sistema de videovigilancia inteligente (Smart NVR) para uso residencial y PyME, desarrollado en **Python 3.12** y **FastAPI**. Diseñado con arquitectura desacoplada para garantizar baja latencia (<500ms en streaming local), bajo consumo de CPU (<10% en reposo por cámara) mediante filtrado híbrido de movimiento en dos fases (OpenCV MOG2 + Deep Learning ligero), grabación de clips MP4 activada por eventos con buffer circular pre/post-roll, despacho asíncrono de alertas por Gmail con instantánea adjunta y un dashboard web interactivo integrado (Single Page Application con Tailwind CSS y HTML5 Canvas).

---

## 📑 Tabla de Contenidos
1. [Arquitectura del Sistema](#-arquitectura-del-sistema)
2. [Características Principales](#-características-principales)
3. [Optimizaciones de Rendimiento (Modo ECO y Lazy Encoding)](#-optimizaciones-de-rendimiento)
4. [Requisitos Previos](#-requisitos-previos)
5. [Instalación y Configuración](#-instalación-y-configuración)
6. [Puesta en Marcha](#-puesta-en-marcha)
7. [Referencia de la API REST](#-referencia-de-la-api-rest)
8. [Guía del Dashboard Web](#-guía-del-dashboard-web)
9. [Configuración de Gmail SMTP (Google App Passwords)](#-configuración-de-gmail-smtp)
10. [Suite de Pruebas Automatizadas](#-suite-de-pruebas-automatizadas)
11. [Licencia](#-licencia)

---

## 🏛 Arquitectura del Sistema

```
[ Fuentes de Video ] (RTSP / Cámaras USB / Archivos MP4 / Simulador Sintético)
        │
        ▼ (Hilos de Ingesta Desacoplados - CameraStream)
[ Motor de Ingesta & Streaming ]
        ├──────────────────────────┬─────────────────────────────┐
        ▼                          ▼                             ▼
[ FrameBroadcaster ]      [ CircularBuffer (3-5s) ]   [ Pipeline Híbrido MOG2 + IA ]
(Lazy JPEG Encoding)      (Deque seguro multihilo)    Fase 1: MOG2 (<10% CPU reposo)
        │                          │                   Fase 2: YOLO / ONNX Runtime CPU
        │                          │                             │
        ▼                          ▼                             ▼
[ Streaming MJPEG ]       [ EventVideoRecorder ] <───────────────┘
(/api/cameras/{id}/stream)(H.264 MP4 con Pre/Post Roll)
        │                          │
        │                          ├─────────────────────────────┐
        ▼                          ▼                             ▼
[ Dashboard Web SPA ]     [ StorageManager ]           [ SQLite Relacional WAL ]
(HTML5, Tailwind CSS)     (Retención, cuotas y purga)  (Eventos, Detecciones, Logs)
                                                                 │
                                                                 ▼
                                                       [ AlertService (Async) ]
                                                       (Gmail SMTP + Cooldown)
```

### Principios de Diseño
1. **Ingesta Desacoplada sin Bloqueo de FPS**: Cada cámara corre en un hilo independiente (`CameraStream`), aislando la captura física de la inferencia neuronal y de los clientes web.
2. **Distribución en Abanico (Fanout) de Codificación Perezosa**: `FrameBroadcaster` comprime a JPEG únicamente cuando hay suscriptores activos escuchando. Con 0 clientes conectados, la compresión JPEG se descarte por completo, reduciendo el overhead a cero.
3. **Pipeline Híbrido en Dos Fases con Escalado Adaptativo**:
   - **Fase 1 (MOG2)**: Sustracción de fondo sobre fotogramas reducidos (320x180) con eliminación de sombras. Si no hay movimiento en la ROI, consume menos del 2% de CPU.
   - **Fase 2 (IA Ligera)**: Al detectarse movimiento dentro de la ROI, se activa YOLOv8n (ONNX Runtime CPU) limitado a 4-6 FPS para clasificar personas y vehículos.
4. **Grabación Continua con Pre/Post-Roll**: Un buffer circular conserva 3 a 5 segundos de video previo a la activación, y el grabador extiende el clip si el movimiento continúa, guardando archivos estándar MP4 con cabeceras `faststart`.
5. **Concurrencia SQLite WAL**: Base de datos local configurada con `PRAGMA journal_mode = WAL` y `busy_timeout = 5000`, permitiendo lecturas concurrentes del servidor web mientras los grabadores escriben eventos.

---

## ⚡ Optimizaciones de Rendimiento

### 🌿 FPS Adaptativo (Modo ECO)
- **Ahorro Dinámico en Reposo**: Cuando una cámara no registra movimiento, no está grabando y no hay usuarios visualizándola en el dashboard, el bucle de procesamiento desciende automáticamente de 15 FPS al modo de reposo ecológico (**4 a 5 FPS**), reduciendo las ejecuciones de MOG2 y descarte de frames en un **~73%**.
- **Detección por Demanda**: Al detectarse movimiento, al activarse una grabación o cuando un usuario abre la transmisión en el navegador, el procesamiento escala instantáneamente a la tasa objetivo máxima (**15 FPS**).
- **Período de Gracia**: Tras cesar la actividad, la cámara se mantiene a 15 FPS durante 3 segundos antes de retornar al modo reposo para prevenir fluctuaciones.
- **Insignia Visual en Vivo**: En la cuadrícula se muestra la insignia dinámica **`🌿 ECO`** reflejando el FPS efectivo en tiempo real.

### 💤 Codificación JPEG Perezosa (Lazy Encoding)
- **Cero Compresión en Bucle Muerto**: Elimina la compresión síncrona innecesaria (`cv2.imencode`) cuando ningún cliente web tiene abierta la vista de streaming.
- **Compresión Bajo Demanda con Caché**: Si se solicita una instantánea (`/api/cameras/{id}/snapshot`) o se conecta un nuevo visor, se ejecuta una única compresión puntual y se guarda en caché hasta que arribe un nuevo fotograma.

---

## 🚀 Características Principales

- **Multi-Fuente**: Compatible con streams RTSP de cámaras IP, webcams USB locales, archivos de video y simulador procedural sintético para pruebas sin hardware físico.
- **Control de Cámaras en Caliente**: Pausa y reactiva cámaras temporalmente desde la interfaz con un solo clic sin reiniciar el servidor.
- **Streaming de Ultrabaja Latencia**: Endpoint MJPEG nativo (`/api/cameras/{id}/stream`) reproducible directamente en cualquier navegador sin complementos ni WebRTC.
- **Reproducción con Desplazamiento (HTTP 206 Partial Content)**: Streaming de video con soporte de cabeceras `Range` para adelantar/retroceder instantáneamente en el reproductor HTML5.
- **Regiones de Interés (ROI) Interactivas**: Editor en canvas para trazar polígonos de exclusión y sliders para calibrar sensibilidad MOG2 y umbrales de confianza en tiempo real.
- **Gestión Avanzada de Almacenamiento**: Eliminación directa de clips individuales y herramienta de purga de archivos huérfanos con liberación automática de espacio en disco.
- **Alertas Inteligentes**: Notificaciones por correo electrónico con instantánea adjunta, cuadro delimitador y período de enfriamiento (*cooldown*) configurable por cámara.
- **Dashboard Web Moderno**: Interfaz Single Page Application (SPA) responsiva construida con Tailwind CSS y JavaScript vanilla, sin necesidad de dependencias de Node.js ni compilación.

---

## 📦 Requisitos Previos

- **Python**: Versión 3.10 o superior (recomendado Python 3.12).
- **FFmpeg**: Opcional pero recomendado para optimización de cabeceras faststart (se incluye soporte automático mediante `imageio-ffmpeg`).
- **Sistema Operativo**: Windows 10/11, Linux (Ubuntu/Debian) o macOS.

---

## ⚙️ Instalación y Configuración

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/tu-usuario/smart-nvr.git
   cd smart-nvr
   ```

2. **Crear y activar un entorno virtual**:
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar variables de entorno**:
   Copie la plantilla de configuración `.env.example`:
   ```bash
   # Windows
   copy .env.example .env

   # Linux / macOS
   cp .env.example .env
   ```
   *(Ajuste los valores de puerto, rutas de almacenamiento o credenciales de Gmail según sus necesidades).*

---

## ▶️ Puesta en Marcha

Inicie el servidor NVR con el script principal:
```bash
python run_server.py
```

O alternativamente mediante Uvicorn:
```bash
uvicorn smart_nvr.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

Acceda a las interfaces del sistema en su navegador:
- **Dashboard Web**: [http://localhost:8000/](http://localhost:8000/)
- **Documentación Interactiva Swagger**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Documentación ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📡 Referencia de la API REST

### Cámaras (`/api/cameras`)
| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/api/cameras` | Lista todas las cámaras con métricas en tiempo real (FPS, modo ECO, suscriptores, alertas). |
| `POST` | `/api/cameras` | Registra una nueva cámara e inicia su pipeline de captura. |
| `GET` | `/api/cameras/{id}` | Obtiene detalles de una cámara específica. |
| `PUT` | `/api/cameras/{id}` | Actualiza configuración de la cámara o alterna su estado de activación/pausa. |
| `DELETE` | `/api/cameras/{id}` | Detiene la cámara, elimina sus registros en BD y purga sus archivos multimedia. |
| `POST` | `/api/cameras/test-connection` | Prueba de conexión no destructiva (RTSP, USB, archivo, sintético). |
| `GET` | `/api/cameras/{id}/snapshot` | Descarga una captura instantánea en formato JPEG (codificada bajo demanda). |
| `GET` | `/api/cameras/{id}/detection-config` | Lee la configuración actual de MOG2, IA y polígonos ROI. |
| `PUT` | `/api/cameras/{id}/detection-config` | Actualización en caliente de sensibilidad, umbral y ROIs sin reinicio. |

### Streaming en Vivo (`/api/cameras/{id}/stream`)
| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/api/cameras/{id}/stream` | Stream MJPEG nativo (`multipart/x-mixed-replace`) de ultrabaja latencia. |

### Incidentes y Eventos (`/api/events`)
| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/api/events` | Consulta paginada con filtros por cámara, fecha, clase de detección y confianza. |
| `GET` | `/api/events/{id}` | Detalle completo del evento con lista de detecciones y logs de alertas. |
| `DELETE` | `/api/events/{id}` | Elimina el evento de la BD y borra físicamente el video MP4 y la imagen del disco. |
| `POST` | `/api/events/purge-orphaned` | Purga del disco clips huérfanos que no corresponden a ningún evento registrado. |
| `POST` | `/api/events/sync-disk` | Sincroniza clips existentes en disco con el historial de eventos de la BD. |
| `GET` | `/api/events/{id}/video` | Streaming del video MP4 con soporte de cabeceras HTTP 206 (Range requests). |
| `GET` | `/api/events/{id}/snapshot` | Descarga la instantánea JPEG anotada con bounding boxes. |

### Configuración del Sistema (`/api/settings`)
| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/api/settings` | Consulta la configuración global con contraseñas enmascaradas. |
| `PUT` | `/api/settings` | Actualiza credenciales SMTP, período de enfriamiento y límites de almacenamiento. |
| `POST` | `/api/settings/test-email` | Despacha un correo de prueba inmediato para validar la conexión SMTP. |

### Salud del Sistema
| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/api/health` | Estado del servidor, cámaras activas, uso de almacenamiento y tiempo de actividad. |
| `GET` | `/api/status` | Resumen detallado de todos los subsistemas del NVR. |

---

## 🖥 Guía del Dashboard Web

El dashboard integrado se divide en 4 vistas intuitivas:

1. **En Vivo (Live Grid)**:
   - Cuadrícula multi-cámara con vista en tiempo real y ajuste automático de columnas.
   - Insignias interactivas de estado ONLINE/PAUSADA, FPS efectivo y modo **`🌿 ECO`**.
   - Efecto visual de pulso ante la presencia confirmada de personas o vehículos.
   - Controles rápidos en cada recuadro para capturar fotogramas, acceder al editor de ROI, pausar/activar o ver en pantalla completa.

2. **Historial de Eventos (Events Gallery)**:
   - Galería de tarjetas con miniaturas, fecha, hora, duración y etiquetas de clasificación.
   - Barra de filtrado avanzado por cámara, rango de fechas, clase de objeto y confianza mínima.
   - Reproductor modal de video HTML5 con soporte de desplazamiento temporal instantáneo (*scrubbing*).
   - Botón de papelera para eliminar clips individuales y botón para purgar archivos huérfanos.

3. **Cámaras & Regiones de Interés (ROI Editor)**:
   - Selector de cámara con carga del fotograma de referencia sobre `<canvas>`.
   - Herramienta interactiva para trazar polígonos delimitando la zona de vigilancia.
   - Deslizadores para calibrar sensibilidad MOG2, área mínima y umbral de confianza de la IA.

4. **Configuración & Alertas**:
   - Formulario para configurar el servidor Gmail SMTP, puerto, usuario, contraseña de aplicación y destinatarios.
   - Ajuste del tiempo de enfriamiento (*cooldown*) para evitar saturación de correos.
   - Botón de prueba de envío de correo con diagnóstico inmediato.
   - Gestión integral de cámaras con asistente de alta y prueba de conexión previa.

---

## 📧 Configuración de Gmail SMTP

Para habilitar el envío de alertas con instantáneas adjuntas mediante Gmail:

1. Inicie sesión en su cuenta de Google y diríjase a **[Seguridad de Google](https://myaccount.google.com/security)**.
2. Asegúrese de tener activada la **Verificación en dos pasos (2FA)**.
3. Ingrese a la sección **[Contraseñas de aplicaciones](https://myaccount.google.com/apppasswords)**.
4. En el campo "Nombre de la aplicación", escriba `Smart NVR` y haga clic en **Crear**.
5. Google generará una clave de 16 caracteres (por ejemplo: `abcd efgh ijkl mnop`).
6. En el archivo `.env` o desde la pestaña de **Configuración** del Dashboard, configure:
   - **Servidor SMTP**: `smtp.gmail.com`
   - **Puerto**: `587`
   - **Usuario**: `su_cuenta@gmail.com`
   - **Contraseña**: La clave de 16 caracteres generada (sin espacios).
   - **Destinatarios**: Correo o lista de correos separados por comas.
7. Presione **Probar Envío de Correo** en el dashboard para verificar la entrega exitosa.

---

## 🧪 Suite de Pruebas Automatizadas

El proyecto incluye una completa suite de pruebas con **pytest** que cubre desde la ingesta física y sintética hasta la API REST y el dashboard:

```bash
# Ejecutar pruebas específicas de Ingesta, API y Dashboard
pytest tests/test_ingestion.py tests/test_api.py tests/test_dashboard.py -v

# Ejecutar la suite completa de pruebas del NVR
pytest -v
```

Todas las pruebas se ejecutan de manera hermética utilizando el simulador sintético procedural y componentes simulados para garantizar 100% de fiabilidad en cualquier entorno.

---

## 📄 Licencia

Este proyecto está bajo la Licencia **MIT**. Consulte el archivo [LICENSE](LICENSE) para más detalles.
