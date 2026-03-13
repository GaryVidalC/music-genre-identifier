# Music Genre Identifier API 
## Versíon del modelo: Metal-v1

API en FastAPI para predecir el genero musical de un archivo `.wav` usando un modelo entrenado.

**IMPORTANTE**: Probablemte solo funciona con .wav y si dura más de 30 segundos deberia crashear. Tamo trabajando pa uste

## Endpoints

- `GET /health`: verifica que la API esta viva.
- `GET /ready`: verifica que modelo, encoder y metadata fueron cargados.
- `GET /model-info`: devuelve la metadata del modelo activo.
- `POST /predict-audio`: recibe un `.wav` y devuelve el genero predicho.

## Ejecutar

```bash
uvicorn app.main:app --reload
```

Documentacion interactiva:

- `http://127.0.0.1:8000/docs`

## Estructura

- `app/main.py`: endpoints y ciclo de vida (`lifespan`).
- `app/model_loader.py`: carga metadata, encoder y modelo.
- `app/audio_processing.py`: extraccion de features de audio.
- `app/inference.py`: prediccion de genero.
- `model/`: artefactos del modelo (`.pkl`, `metadata.json`).
