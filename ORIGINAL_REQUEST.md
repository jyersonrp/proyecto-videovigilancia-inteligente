# Original User Request

## Initial Request — 2026-09-06T21:29:52Z

Sistema de Videovigilancia Inteligente (Smart NVR) modular para uso residencial o pyme, con ingesta multi-stream desacoplada, pipeline híbrido de detección (sustracción de fondo MOG2 con OpenCV y confirmación por IA ligera para personas/vehículos), servidor FastAPI con streaming en vivo, almacenamiento de clips activado por eventos con buffer pre/post-roll, despacho de alertas por Gmail (SMTP seguro) con snapshot adjunto y dashboard web interactivo integrado.

Working directory: c:\Users\jyers\Downloads\proyecto Sist Videovigilancia Inteligente
Integrity mode: demo

## Requirements

### R1. Ingesta y Detección Híbrida Inteligente
- Captura de streams de video (RTSP, cámaras web locales o archivos de video) ejecutada en hilos/procesos independientes desacoplados para garantizar fluidez sin bloqueo de la tasa de frames (FPS).
- Pipeline híbrido en dos fases de bajo consumo:
  1. Detección previa de movimiento en reposo utilizando sustracción de fondo con OpenCV (MOG2) para filtrar inactividad, sombras leves o ruido estático.
  2. Al activarse movimiento, disparar inferencia ligera de Deep Learning (ej. modelo YOLOv8n u ONNX Runtime compatible con CPU) para clasificar y confirmar presencia de personas o vehículos.
- Soporte para definición de Regiones de Interés (ROIs) configurables por cámara.

### R2. Servidor NVR y Streaming con FastAPI
- API REST asíncrona implementada con FastAPI que permita:
  - Gestionar cámaras (agregar, editar, eliminar y consultar estado de conexión).
  - Configurar umbrales de sensibilidad de movimiento y confianza de detección.
  - Consultar historial paginado de eventos e incidentes detectados.
- Endpoint de streaming en vivo de baja latencia (MJPEG / streaming multipart o WebSockets) accesible directamente desde el navegador web.

### R3. Sistema de Alertas por Correo (Gmail SMTP)
- Servicio notificador asíncrono y desacoplado, configurado para envío de correos mediante Gmail con autenticación segura (SMTP sobre TLS/SSL utilizando App Passwords de Google).
- Mensaje de alerta estructurado con: fecha y hora exacta, identificador de la cámara, objeto/clase detectada y fotografía snapshot con bounding boxes adjunta.
- Control de frecuencia y enfriamiento (cooldown period configurable) por cámara para prevenir inundación de correos ante movimiento continuo.
- Arquitectura desacoplada con interfaz modular para permitir agregar canales de mensajería adicionales (como WhatsApp o Webhooks) en el futuro.

### R4. Almacenamiento y Grabación de Video por Eventos
- Grabación de clips de video en formato reproducible estándar (MP4 / H.264) activada exclusivamente al confirmar una detección válida.
- Mecanismo de buffer circular en memoria para incluir pre-roll (3 a 5 segundos previos al evento) y post-roll (5 a 10 segundos posteriores), garantizando la captura del incidente completo sin malgastar almacenamiento en disco.
- Registro relacional en base de datos local SQLite con los metadatos de cada evento (identificador único, marca temporal, cámara, etiquetas de detección, ruta relativa al clip de video y al snapshot).

### R5. Dashboard Web Integrado
- Interfaz web responsiva servida directamente por FastAPI (HTML5, Tailwind CSS y JavaScript nativo, sin dependencias complejas de compilación frontend).
- Panel principal con cuadrícula de visualización en vivo de los streams activos.
- Sección de historial de eventos con galería de miniaturas filtrables por fecha/cámara y reproductor de video HTML5 integrado para revisar los clips grabados.
- Vista de configuración para ajustar parámetros de alerta y credenciales de correo vía `.env`.

### R6. Entorno de Simulación y Suite de Verificación Automatizada
- Módulo generador de video sintético / simulador de cámara para reproducir secuencias de prueba (frames con movimiento controlado de figuras/personas simuladas) que permita ejecutar y verificar el flujo completo de detección sin requerir hardware de cámara física.
- Suite de pruebas automatizadas (pytest) que valide:
  - Ingesta y generación continua de frames sin pérdida de memoria.
  - Activación del filtro MOG2 y disparo de inferencia ante movimiento.
  - Generación de clips con buffer pre/post-roll e inserción en SQLite.
  - Despacho de alerta por correo (con mock SMTP para pruebas locales).
  - Disponibilidad de endpoints de la API y renderizado del dashboard web.

## Acceptance Criteria

### Ingesta y Detección
- [ ] El sistema ingesta streams de video de forma desacoplada y mantiene la transmisión fluida sin congelamientos ante cargas de inferencia.
- [ ] La sustracción de fondo con MOG2 ignora fotogramas estáticos con uso mínimo de CPU (< 10% en reposo por cámara).
- [ ] Ante movimiento en la zona de interés, el detector clasifica correctamente personas/vehículos y descarta falsos positivos por debajo del umbral de confianza configurado.

### Servidor y API FastAPI
- [ ] FastAPI levanta sin errores y expone la documentación interactiva en `/docs`.
- [ ] El endpoint de streaming de video entrega el feed continuo en vivo a un cliente HTTP/navegador con latencia inferior a 500ms en red local.
- [ ] Los endpoints REST permiten consultar eventos con filtros por fecha, tipo de detección y cámara.

### Grabación y Base de Datos
- [ ] Los clips grabados ante eventos se guardan en formato MP4 reproducible en cualquier navegador moderno.
- [ ] Cada clip incluye los fotogramas del buffer previos a la activación (pre-roll) y el lapso posterior (post-roll).
- [ ] Cada detección genera una fila en la base de datos SQLite con timestamp, cámara, clase y rutas válidas a sus archivos multimedia.

### Alertas y Notificaciones
- [ ] Se genera un correo con formato visual claro y snapshot de evidencia adjunto al confirmarse una detección válida.
- [ ] Múltiples detecciones consecutivas dentro de la ventana de enfriamiento (cooldown) no provocan envíos redundantes de correo.
- [ ] Las credenciales y configuraciones del servidor SMTP se leen de forma segura desde variables de entorno (`.env`).

### Dashboard y Usabilidad
- [ ] El dashboard web en la ruta raíz `/` muestra el estado de las cámaras y los feeds en vivo.
- [ ] La interfaz permite hacer clic en un evento del historial y reproducir su clip MP4 directamente en el navegador.

### Calidad y Tests
- [ ] La suite de pruebas de `pytest` ejecuta pruebas de extremo a extremo utilizando el simulador sintético y pasa con 100% de éxito.
- [ ] El proyecto incluye `README.md` con instrucciones claras de instalación (`requirements.txt`), configuración (`.env.example`) y puesta en marcha con un solo comando.
