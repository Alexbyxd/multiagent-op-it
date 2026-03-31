"""Setup de base de datos SQLite con datos de prueba."""
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

from src.config import settings


def create_tables(conn: sqlite3.Connection) -> None:
    """Crea las tablas necesarias."""
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            error_code TEXT,
            solution TEXT NOT NULL,
            severity TEXT NOT NULL,
            created_at DATETIME NOT NULL,
            resolved_at DATETIME NOT NULL
        )
    """)
    conn.commit()


def insert_sample_tickets(conn: sqlite3.Connection) -> None:
    """Inserta tickets de prueba."""
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM tickets")
    if cursor.fetchone()[0] > 0:
        return
    
    tickets = [
        (
            "Error 503 en servidor web de producción",
            "Los usuarios reportan error 503 Service Unavailable al acceder al sitio web principal. El servidor web devuelve error inmediatamente.",
            "503",
            "Reiniciar el servicio nginx: sudo systemctl restart nginx. Verificar logs en /var/log/nginx/error.log. Aumentar worker_connections en nginx.conf.",
            "critical",
            datetime.now() - timedelta(days=5),
            datetime.now() - timedelta(days=5)
        ),
        (
            "Conexión lenta a base de datos PostgreSQL",
            "Las consultas a la base de datos tardan más de 10 segundos. Los usuarios experimentan timeouts frecuentes.",
            "DB_TIMEOUT",
            "Ejecutar ANALYZE en las tablas afectadas. Aumentar shared_buffers en postgresql.conf. Revisar índices faltantes en consultas lentas.",
            "high",
            datetime.now() - timedelta(days=10),
            datetime.now() - timedelta(days=8)
        ),
        (
            "Fallo de autenticación con LDAP",
            "Usuarios no pueden iniciar sesión. El servidor LDAP responde con error de timeout.",
            "LDAP_TIMEOUT",
            "Verificar conectividad con servidor LDAP: ldapsearch -h ldap.server.com. Revisar configuración de timeout en /etc/ldap/ldap.conf. Reiniciar servicio slapd.",
            "critical",
            datetime.now() - timedelta(days=3),
            datetime.now() - timedelta(days=2)
        ),
        (
            "API Gateway devuelve 502 Bad Gateway",
            "El API gateway no puede comunicarse con los microservicios backend. Todos los endpoints fallan.",
            "502",
            "Verificar que los servicios backend estén corriendo: docker ps. Revisar configuración de upstream en nginx. Reiniciar contenedores affected.",
            "high",
            datetime.now() - timedelta(days=7),
            datetime.now() - timedelta(days=6)
        ),
        (
            "Espacio en disco bajo en servidor de logs",
            "El servidor tiene menos de 5% de espacio disponible. Los servicios comienzan a fallar.",
            "DISK_FULL",
            "Ejecutar du -sh /var/log/* para identificar directorios grandes. Rotar logs antiguos: logrotate -f /etc/logrotate.conf. Eliminar archivos temporales.",
            "medium",
            datetime.now() - timedelta(days=2),
            datetime.now() - timedelta(days=1)
        ),
        (
            "Error de certificado SSL expirado",
            "Los navegadores muestran advertencia de certificado no válido. La API retorna errores de conexión segura.",
            "SSL_EXPIRED",
            "Renovar certificado: certbot renew --nginx. Verificar fecha de expiración: openssl s_client -connect domain.com:443. Actualizar DNS si es necesario.",
            "high",
            datetime.now() - timedelta(days=1),
            datetime.now()
        ),
    ]
    
    cursor.executemany(
        "INSERT INTO tickets (title, description, error_code, solution, severity, created_at, resolved_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        tickets
    )
    conn.commit()


def setup_database() -> None:
    """Inicializa la base de datos."""
    settings.DATA_DIR.mkdir(exist_ok=True)
    
    conn = sqlite3.connect(settings.DB_PATH)
    create_tables(conn)
    insert_sample_tickets(conn)
    conn.close()
    
    print(f"Base de datos inicializada en {settings.DB_PATH}")


if __name__ == "__main__":
    setup_database()