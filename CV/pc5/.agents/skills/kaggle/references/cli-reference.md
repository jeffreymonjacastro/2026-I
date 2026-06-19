# Kaggle CLI Reference

## Principios

- Usar `kaggle --version` primero para confirmar que la CLI existe.
- No imprimir ni guardar tokens, `kaggle.json`, cookies ni credenciales.
- En Kaggle CLI, los notebooks se gestionan con el grupo `kaggle kernels`.
- Pasar slugs en formato `owner/slug` para kernels, datasets y modelos cuando el comando lo pida.
- Ejecutar notebooks desde la carpeta que contiene `kernel-metadata.json` y el archivo indicado por `code_file`.

## Autenticacion y ayuda

```powershell
kaggle --version
kaggle --help
kaggle kernels --help
kaggle datasets --help
kaggle competitions --help
kaggle models --help
```

## Listar notebooks y kernels

```powershell
# Mis notebooks/kernels
kaggle kernels list --mine

# Solo notebooks
kaggle kernels list --mine --kernel-type notebook

# Solo scripts
kaggle kernels list --mine --kernel-type script

# Buscar por texto
kaggle kernels list --mine --search "texto"

# Listar kernels de un usuario
kaggle kernels list --user <username>

# Listar kernels asociados a dataset o competition
kaggle kernels list --dataset <owner/dataset-slug>
kaggle kernels list --competition <competition-slug>
```

Opciones utiles:

```powershell
kaggle kernels list --mine --page-size 100
kaggle kernels list --mine --sort-by dateRun
kaggle kernels list --mine --csv
```

## Descargar o inspeccionar kernels

```powershell
# Descargar codigo del kernel
kaggle kernels pull <owner/kernel-slug> -p <folder> --metadata

# Listar archivos de output del ultimo run
kaggle kernels files <owner/kernel-slug>

# Estado del ultimo run
kaggle kernels status <owner/kernel-slug>

# Logs del ultimo run
kaggle kernels logs <owner/kernel-slug>
kaggle kernels logs <owner/kernel-slug> --follow --interval 5
```

## Ejecutar notebooks/kernels

Preparar carpeta:

```text
kaggle/
└── <vn>/
    ├── input/
    │   ├── kernel-metadata.json
    │   └── main.ipynb
    └── outputs/
```

Ejecutar desde `kaggle/<vn>/input/`:

```powershell
cd kaggle\<vn>\input
kaggle kernels push -p . --accelerator NvidiaTeslaT4
```

Sin GPU especifica:

```powershell
kaggle kernels push -p .
```

Con timeout:

```powershell
kaggle kernels push -p . --accelerator NvidiaTeslaT4 --timeout 3600
```

Esperar y descargar outputs:

```powershell
kaggle kernels status <owner/kernel-slug>
kaggle kernels output <owner/kernel-slug> -p ..\outputs --force
```

Para logs:

```powershell
kaggle kernels logs <owner/kernel-slug>
```

## Metadata de kernel

Para cargar y ejecutar el codigo, el archivo `kernel-metadata.json` debe ser especificado.

Una plantilla de ejemplo para este archivo se encuentra en `./kernel-metadata-template.json`.

```json
{
  "id": "owner/kernel-slug",
  "title": "kernel-title",
  "code_file": "main.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

Puedes usar `kaggle kernels init -p <folder>` para que la API cree este archivo para un kernel nuevo. Para obtener metadata de un kernel existente, usa `kaggle kernels pull <owner/kernel-slug> -p <folder> --metadata`.

### Campos

- `id`: slug del kernel en formato `owner/kernel-slug`.
- `id_no`: ID numerico del kernel. Si se proporciona con `id`, Kaggle prefiere `id_no`.
- `title`: titulo del kernel. Requerido para kernels nuevos.
- `code_file`: ruta del codigo fuente, relativa a `kernel-metadata.json` si no es absoluta.
- `language`: `python`, `r` o `rmarkdown`.
- `kernel_type`: `script` o `notebook`.
- `is_private`: privacidad del kernel. Por defecto, `true`.
- `enable_gpu`: si usa GPU. Por defecto, `false`.
- `enable_internet`: si usa internet. Por defecto, `false`.
- `machine_shape`: acelerador/GPU, por ejemplo `NvidiaTeslaT4`, `NvidiaTeslaP100` o `Tpu1VmV38`.
- `dataset_sources`: datasets en formato `owner/dataset-slug`.
- `competition_sources`: competencias en formato `competition-slug`.
- `kernel_sources`: kernels en formato `owner/kernel-slug`.
- `model_sources`: modelos en formato `owner/model-slug/framework/variation-slug/version-number`.

## Datasets

```powershell
# Mis datasets
kaggle datasets list --mine

# Buscar datasets
kaggle datasets list --search "texto"

# Datasets de usuario
kaggle datasets list --user <username>

# Ver archivos
kaggle datasets files <owner/dataset-slug>

# Descargar dataset completo
kaggle datasets download <owner/dataset-slug> -p <folder> --unzip

# Descargar archivo especifico
kaggle datasets download <owner/dataset-slug> -f <file-name> -p <folder>
```

Crear dataset:

```powershell
kaggle datasets init -p <folder>
kaggle datasets create -p <folder>
kaggle datasets create -p <folder> --dir-mode zip
kaggle datasets version -p <folder> -m "mensaje de version"
```

Para crear un dataset nuevo, la carpeta debe contener un archivo `dataset-metadata.json` en la raiz. Una plantilla de ejemplo se encuentra en `./dataset-metadata-template.json`; copiala al staging del dataset con el nombre exacto `dataset-metadata.json`.

Ejemplo minimo:

```json
{
  "title": "dataset-title",
  "id": "owner/dataset-slug",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ],
  "subtitle": "Short dataset subtitle",
  "description": "Describe the dataset contents, source, intended use, folder structure, and any important limitations."
}
```

Notas practicas:

- `id` debe usar formato `owner/dataset-slug`.
- La CLI espera `dataset-metadata.json` en singular.
- Si el dataset contiene subcarpetas, usar `--dir-mode zip` para subirlas como directorios comprimidos; sin ese flag, la CLI puede omitir directorios segun el modo usado.
- `kaggle datasets create` crea el dataset inicial. Para actualizar un dataset existente, usar `kaggle datasets version -p <folder> -m "mensaje de version"`.

## Competitions

```powershell
kaggle competitions list
kaggle competitions files <competition-slug>
kaggle competitions download <competition-slug> -p <folder>
kaggle competitions submissions <competition-slug>
kaggle competitions submit <competition-slug> -f <submission-file> -m "mensaje"
```

## Models

```powershell
kaggle models list
kaggle models get <owner/model-slug> -p <folder>
kaggle models init -p <folder>
kaggle models create -p <folder>
```

## Fuentes oficiales

- Kaggle CLI kernels: `https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md`
- Kaggle kernel metadata: `https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md`
- Kaggle CLI datasets: `https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md`
- Kaggle dataset metadata: `https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md`
- Kaggle CLI competitions: `https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md`
- Kaggle CLI models: `https://github.com/Kaggle/kaggle-cli/blob/main/docs/models.md`
