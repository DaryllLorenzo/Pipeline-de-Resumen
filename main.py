from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

from api.routes import router as api_router
from api.models import ErrorResponse

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="Advanced Summarization Pipeline API",
    description="API para resumen extractivo avanzado multilingüe usando scikit-learn",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Manejo global de excepciones
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Excepción global no manejada: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Error interno del servidor",
            details={"exception": str(exc)},
            code=500
        ).dict()
    )

# Incluir rutas
app.include_router(api_router)

# Ruta raíz
@app.get("/")
async def root():
    return {
        "message": "Advanced Summarization Pipeline API",
        "version": "1.0.0",
        "endpoints": {
            "documentation": "/docs",
            "health": "/api/v1/health",
            "summarize": "/api/v1/summarize",
            "batch_summarize": "/api/v1/summarize/batch",
            "compare": "/api/v1/metrics/compare",
            "config": "/api/v1/config",
            "service_metrics": "/api/v1/metrics/service"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )