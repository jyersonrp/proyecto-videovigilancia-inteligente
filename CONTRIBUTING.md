# Guía de Contribución a Smart NVR 🤝

¡Gracias por tu interés en contribuir a **Smart NVR**! Este proyecto tiene como objetivo ofrecer una solución de videovigilancia inteligente, modular y de alto rendimiento accesible para entornos residenciales y pequeñas empresas.

---

## 🛠️ Cómo Empezar

### 1. Fork y Clonado
1. Haz un **Fork** de este repositorio en GitHub.
2. Clona tu fork localmente:
   ```bash
   git clone https://github.com/TU-USUARIO/proyecto-videovigilancia-inteligente.git
   cd proyecto-videovigilancia-inteligente
   ```

### 2. Configurar Entorno de Desarrollo
Recomendamos utilizar Python 3.12 con un entorno virtual aislado:

```bash
# Crear entorno virtual
python -m venv .venv

# Activar en Windows
.\.venv\Scripts\Activate.ps1

# Activar en Linux / macOS
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

---

## 🧪 Pruebas Automatizadas

Antes de enviar un Pull Request, asegúrate de que todas las pruebas pasen satisfactoriamente:

```bash
# Pruebas principales (Ingesta, API y Dashboard)
pytest tests/test_ingestion.py tests/test_api.py tests/test_dashboard.py -v

# Opcional: Ejecutar con reporte de cobertura
pytest --durations=10
```

---

## 📋 Reglas y Estándares de Código

- **Tipado estricto**: Todo el código nuevo debe incluir type hints (`typing` / `annotations`).
- **Arquitectura desacoplada**: Los hilos de captura de video nunca deben bloquearse por inferencias de IA ni por conexiones lentas de clientes web.
- **Rendimiento primero**:
  - Evitar copias innecesarias de matrices `np.ndarray`.
  - Asegurar que la compresión JPEG sea perezosa (*lazy encoding*) y no ocurra en bucles cerrados si no hay suscriptores.
- **Seguridad**:
  - No exponer contraseñas en logs ni en respuestas de endpoints REST.
  - Los endpoints de configuración deben mantener credenciales enmascaradas.

---

## 🚀 Envío de Pull Requests

1. Crea una rama descriptiva para tu cambio:
   ```bash
   git checkout -b feature/nombre-de-la-mejora
   ```
2. Realiza tus commits con mensajes claros y descriptivos (recomendamos Conventional Commits: `feat: ...`, `fix: ...`, `docs: ...`, `perf: ...`).
3. Sube tus cambios a tu fork:
   ```bash
   git push origin feature/nombre-de-la-mejora
   ```
4. Abre un **Pull Request** hacia la rama `main` del repositorio original describiendo el problema y la solución.

¡Agradecemos enormemente tus contribuciones para hacer de Smart NVR el mejor sistema de videovigilancia de código abierto!
