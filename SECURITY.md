# Política de Seguridad 🔒

La seguridad de las transmisiones de video, grabaciones y credenciales del sistema es una prioridad fundamental en **Smart NVR**.

---

## 🛡️ Versiones Soportadas

Actualmente se brinda soporte de seguridad activo a la rama principal:

| Versión | Soportada          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

---

## 🚨 Reporte de Vulnerabilidades

Si descubres una posible vulnerabilidad de seguridad en este proyecto:

1. **NO abras un issue público.**
2. Envía un correo electrónico privado o utiliza la función de **Security Advisory privada de GitHub**:
   - Pestaña **Security** > **Report a vulnerability**.
3. Incluye en tu reporte:
   - Tipo de vulnerabilidad y vector de ataque.
   - Pasos detallados para reproducir el fallo.
   - Impacto potencial y soluciones sugeridas si las tienes.

Nos comprometemos a revisar y responder a los reportes de seguridad en un plazo máximo de 48 horas.

---

## 🔒 Buenas Prácticas Recomendadas para Despliegues

- **Segmentación de Red**: Ubica las cámaras IP y el servidor NVR en una VLAN dedicada y aislada del tráfico de invitados.
- **Credenciales Fuertes**: Nunca utilices las contraseñas predeterminadas del fabricante en tus cámaras IP o RTSP.
- **Acceso Remoto**: Si requieres acceso fuera de tu red local, utiliza una VPN segura (como WireGuard o Tailscale) o un túnel cifrado inverso con autenticación fuerte en lugar de abrir puertos directamente en tu router.
