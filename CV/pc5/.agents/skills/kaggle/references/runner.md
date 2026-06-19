# Kaggle Runner

## Uso

Usar esta referencia para preparar, subir, ejecutar y recuperar notebooks o scripts en Kaggle con Kaggle CLI dentro de carpetas versionadas.

## Estructura esperada

```text
kaggle/
└── <vn>/
    ├── input/
    │   ├── kernel-metadata.json
    │   └── <notebook>.ipynb o <script>.py
    └── outputs/
```

`<vn>` es la version local del experimento, por ejemplo `v1`, `v2` o `v3`.

## Reglas

- No guardar `kaggle.json`, tokens ni credenciales en el repo.
- No imprimir secretos durante validacion de autenticacion.
- Mantener una version por carpeta.
- Ejecutar desde `kaggle/<vn>/input/`.
- Descargar outputs remotos en `kaggle/<vn>/outputs/`.
- Usar GPU T4 por defecto: `--accelerator NvidiaTeslaT4`.
- Mantener `enable_gpu: true` en `kernel-metadata.json` cuando se use GPU.
- Usar `kernel_type: "notebook"` para `.ipynb` y `kernel_type: "script"` para `.py`.

## Estructura de notebooks

- No poner todo el codigo en un solo chunk/celda.
- Dividir el notebook en bloques pequenos y ejecutables de forma incremental.
- Antes de cada bloque de codigo, agregar una celda Markdown con una subseccion que explique el objetivo del bloque.
- Usar nombres de celdas o ids estables cuando se edite JSON de notebook.
- Mantener outputs impresos cortos: rutas, conteos, estado y errores resumidos.

Estructura recomendada:

```text
1. Markdown: objetivo del notebook
2. Markdown: imports y configuracion
3. Code: imports
4. Markdown: resolver paths y parametros
5. Code: paths, seeds, carpetas de output
6. Markdown: validar entradas
7. Code: checks de datasets/archivos
8. Markdown: procesar etapa 1
9. Code: trabajo + checkpoint parcial
10. Markdown: procesar etapa 2
11. Code: trabajo + checkpoint parcial
12. Markdown: escribir outputs finales
13. Code: resumen, CSV/JSON/modelos/imagenes
```

## Persistencia parcial

Kaggle no preserva `/kaggle/working` si el run falla antes de publicar una nueva version de outputs. Para no perder trabajo largo:

- Escribir archivos parciales en `/kaggle/working` despues de cada etapa cara.
- Actualizar `run_summary.json` o `progress.json` con estado, etapa actual, conteos, rutas y error si existe.
- Guardar CSV/JSON/checkpoints incrementalmente, no solo al final.
- Preferir que el notebook termine con estado remoto `COMPLETE` y un JSON interno con `"status": "failed"` si se necesita recuperar outputs parciales.
- Solo relanzar la excepcion si el objetivo explicito es hacer fallar el kernel y no importa perder outputs.
- Limpiar outputs locales antes de descargar nuevos resultados para evitar confundir archivos viejos con nuevos.

Patron recomendado:

```python
import json
import traceback
from pathlib import Path

output_dir = Path("/kaggle/working")
progress_path = output_dir / "run_summary.json"
progress = {"status": "running", "stage": "start", "artifacts": []}

def save_progress():
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

try:
    progress["stage"] = "load_inputs"
    save_progress()
    # cargar datos

    progress["stage"] = "process"
    save_progress()
    # trabajo caro
    # escribir checkpoint parcial aqui

    progress["stage"] = "write_outputs"
    # escribir CSV/JSON/modelos/imagenes finales

    progress["status"] = "complete"
    progress["stage"] = "done"
except Exception as exc:
    progress["status"] = "failed"
    progress["error"] = repr(exc)
    progress["traceback"] = traceback.format_exc()
finally:
    save_progress()
```

Si se necesita recuperar outputs aun con error, no volver a hacer `raise` despues del `finally`; dejar que el notebook termine y revisar `run_summary.json`.

## Pipeline CLI

1. Confirmar CLI:

```powershell
kaggle --version
```

2. Validar carpeta:

```powershell
Get-ChildItem kaggle\<vn>\input
```

La carpeta debe contener `kernel-metadata.json` y el archivo indicado por `code_file`.

3. Ejecutar kernel desde `input/`:

```powershell
cd kaggle\<vn>\input
kaggle kernels push -p . --accelerator NvidiaTeslaT4
```

4. Esperar estado remoto:

```powershell
kaggle kernels status <owner/kernel-slug>
```

Esperar hasta `COMPLETE`. Si termina en `ERROR`, revisar logs; no asumir que outputs parciales existen.

5. Revisar logs si hace falta:

```powershell
kaggle kernels logs <owner/kernel-slug>
```

6. Descargar outputs:

```powershell
New-Item -ItemType Directory -Force ..\outputs | Out-Null
kaggle kernels output <owner/kernel-slug> -p ..\outputs --force
```

7. Validar resultado local:

```powershell
Get-ChildItem ..\outputs
```

## Reporte final

Reportar:

- kernel slug usado
- estado final remoto
- ruta local de outputs
- archivos descargados
- evidencia clave de logs o resultados
- si el notebook uso `run_summary.json`, reportar `status`, `stage` y `error`

Si falla autenticacion, pedir al usuario reparar credenciales de Kaggle sin solicitar ni imprimir tokens.
