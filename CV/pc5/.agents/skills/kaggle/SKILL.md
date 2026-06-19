---
name: kaggle
description: Skill global para trabajar con Kaggle CLI. Usar cuando Codex deba listar notebooks/kernels/datasets/modelos/competencias, preparar metadata de kernels, ejecutar notebooks o scripts remotos en Kaggle, monitorear status/logs y descargar outputs. Carga references/runner.md para ejecucion end-to-end y references/cli-reference.md para comandos y metadata.
---

# kaggle

## Router

- Para ejecutar un notebook o script remoto en Kaggle, leer `references/runner.md`.
- Para consultar comandos, flags, metadata, datasets, competitions, kernels o models, leer `references/cli-reference.md`.
- Para tareas mixtas, leer primero `references/runner.md`; si falta un comando o campo de metadata, leer `references/cli-reference.md`.

## Reglas comunes

- No guardar `kaggle.json`, tokens ni credenciales en repositorios.
- No imprimir secretos ni contenido de credenciales.
- Confirmar CLI con `kaggle --version` antes de ejecutar flujos.
- En Kaggle CLI, notebooks y scripts se gestionan como `kernels`.
- Usar slugs `owner/slug` cuando Kaggle CLI pida identificadores remotos.
- Para ejecuciones versionadas en repos, preferir `kaggle/<vn>/input/` y `kaggle/<vn>/outputs/`.

## Handoff

Al terminar una ejecucion remota, reportar:

- kernel slug
- estado final
- ruta local de outputs
- archivos descargados
- evidencia clave de logs o resultados
