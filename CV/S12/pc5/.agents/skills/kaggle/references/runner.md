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

Esperar hasta `COMPLETE`. Si termina en error, revisar logs.

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

Si falla autenticacion, pedir al usuario reparar credenciales de Kaggle sin solicitar ni imprimir tokens.
