# Despliegue en Google Cloud

Esta guía explica cómo se despliega la API, qué recursos existen y cómo
operarlos sin tener que reconstruir mentalmente todo el proceso. También sirve
como referencia para cualquier agente que trabaje en el repositorio.

Estado comprobado el 29 de septiembre de 2026.

## Resumen rápido

La API corre en Cloud Run. Las imágenes Docker se almacenan en Artifact
Registry y GitHub Actions hace el despliegue automáticamente después de que el
CI termina correctamente.

```text
push a main
    -> CI: lint, tests y Docker build
    -> autenticación de GitHub en Google Cloud
    -> build y push de la imagen con el SHA del commit
    -> nueva revisión de Cloud Run
    -> GET /health
```

No hay una máquina virtual encendida permanentemente. Cloud Run puede reducir
el servicio a cero instancias cuando no recibe tráfico.

## Recursos actuales

| Recurso | Valor |
|---|---|
| Proyecto de Google Cloud | `gen-lang-client-0620673955` |
| Número del proyecto | `845382797070` |
| Región | `southamerica-west1` |
| Servicio de Cloud Run | `music-genre-identifier` |
| Artifact Registry | `music-genre-identifier` |
| Cuenta usada por GitHub | `github-deployer@gen-lang-client-0620673955.iam.gserviceaccount.com` |
| Cuenta usada por Cloud Run | `845382797070-compute@developer.gserviceaccount.com` |
| Workload Identity Pool | `github-actions` |
| Proveedor OIDC | `github` |
| Rama desplegada | `main` |

URL actual del servicio:

```text
https://music-genre-identifier-j7ngpo2fqq-tl.a.run.app
```

La URL debe consultarse nuevamente si el servicio se elimina y se crea otra
vez:

```bash
gcloud run services describe music-genre-identifier \
  --project=gen-lang-client-0620673955 \
  --region=southamerica-west1 \
  --format='value(status.url)'
```

### Configuración actual de Cloud Run

| Opción | Valor |
|---|---:|
| CPU | 1 |
| Memoria | 2 GiB |
| Concurrencia | 1 request por instancia |
| Instancias mínimas | 0 |
| Instancias máximas | 1 |
| CPU fuera de requests | Desactivada |
| Startup CPU boost | Activado |
| Tráfico | 100% a la revisión más reciente |
| Acceso | Público mediante `allUsers` + `roles/run.invoker` |

`min=0` permite escalar a cero. Esto reduce el uso en reposo, pero la primera
petición después de un periodo sin tráfico puede tardar más por el cold start.
El servicio sigue existiendo aunque no haya ninguna instancia activa.

## Qué ocurre durante un deploy

Los dos workflows tienen responsabilidades distintas:

- `.github/workflows/ci.yml` valida cada push y pull request.
- `.github/workflows/deploy.yml` despliega únicamente después de un CI exitoso
  originado por un push a `main`.

El CD realiza estos pasos:

1. Obtiene el SHA exacto que fue probado por CI.
2. Hace checkout de ese commit, no de otro commit más reciente.
3. GitHub obtiene una credencial temporal de Google Cloud mediante OIDC y
   Workload Identity Federation.
4. Docker inicia sesión en Artifact Registry con esa credencial temporal.
5. Buildx construye la imagen y la publica con el SHA como tag.
6. Cloud Run crea una revisión nueva con esa imagen.
7. Cloud Run dirige el tráfico a la revisión nueva cuando está lista.
8. El workflow llama a `/health`. Un error HTTP hace fallar el CD.

La imagen tiene esta forma:

```text
southamerica-west1-docker.pkg.dev/
  gen-lang-client-0620673955/
  music-genre-identifier/
  music-genre-identifier:<COMMIT_SHA>
```

Aunque el repositorio y la imagen se llaman igual, son elementos distintos:

```text
REGIÓN-docker.pkg.dev/PROYECTO/REPOSITORIO/IMAGEN:TAG
```

El CI también construye una imagen, pero solo para comprobar que el Dockerfile
funciona. El CD vuelve a construirla y publica la versión que se desplegará.

## Variables de GitHub

Se administran en:

```text
GitHub repository
  -> Settings
  -> Secrets and variables
  -> Actions
  -> Variables
```

| Variable | Valor actual | Uso |
|---|---|---|
| `GCP_PROJECT_ID` | `gen-lang-client-0620673955` | Proyecto que contiene todos los recursos |
| `GCP_REGION` | `southamerica-west1` | Región de Artifact Registry y Cloud Run |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | `projects/845382797070/locations/global/workloadIdentityPools/github-actions/providers/github` | Permite que GitHub solicite identidad a Google |
| `GCP_SERVICE_ACCOUNT` | `github-deployer@gen-lang-client-0620673955.iam.gserviceaccount.com` | Identidad que ejecuta el deploy |
| `GAR_REPOSITORY` | `music-genre-identifier` | Repositorio de imágenes Docker |
| `CLOUD_RUN_SERVICE` | `music-genre-identifier` | Servicio actualizado por el CD |

Estas variables no son contraseñas. La credencial de acceso se genera durante
el workflow, dura poco tiempo y no se guarda en GitHub.

Cambiar una variable no crea el recurso correspondiente:

- Si cambia `GAR_REPOSITORY`, hay que crear el repositorio nuevo y otorgar
  `roles/artifactregistry.writer` sobre él.
- Si cambia `CLOUD_RUN_SERVICE`, hay que crear o autorizar el servicio nuevo.
  El permiso actual de Cloud Run está limitado al servicio existente.
- Si cambia el proyecto o la región, hay que recrear los recursos y revisar
  todos los permisos IAM.
- Si cambia la cuenta o el proveedor de identidad, también deben actualizarse
  los bindings de Workload Identity Federation.

Una variable inexistente se convierte en una cadena vacía en GitHub Actions.
Por eso un nombre mal escrito suele producir un URI inválido o un error de
autenticación.

## Autenticación e IAM

No se usa una clave JSON de larga duración. GitHub presenta un token OIDC y
Google permite asumir temporalmente la identidad de `github-deployer`.

El proveedor acepta solamente tokens que cumplan ambas condiciones:

```text
repository == GaryVidalC/music-genre-identifier
ref == refs/heads/main
```

Permisos actuales:

| Identidad | Recurso | Rol | Motivo |
|---|---|---|---|
| GitHub mediante WIF | `github-deployer` | `roles/iam.workloadIdentityUser` | Asumir temporalmente la cuenta de deploy |
| `github-deployer` | Artifact Registry `music-genre-identifier` | `roles/artifactregistry.writer` | Publicar imágenes |
| `github-deployer` | Cloud Run `music-genre-identifier` | `roles/run.developer` | Crear revisiones y actualizar el servicio |
| `github-deployer` | Cuenta de ejecución de Cloud Run | `roles/iam.serviceAccountUser` | Desplegar usando esa identidad de runtime |
| `allUsers` | Cloud Run `music-genre-identifier` | `roles/run.invoker` | Permitir acceso público |

El workflow necesita además:

```yaml
permissions:
    contents: read
    id-token: write
```

`contents: read` permite hacer checkout. `id-token: write` permite solicitar el
token OIDC. No significa que el workflow pueda modificar el repositorio.

## Cómo desplegar normalmente

No es necesario ejecutar comandos de Google Cloud para un deploy normal:

```bash
git add .
git commit -m "Describe the change"
git push origin main
```

Después hay que revisar la pestaña **Actions** de GitHub:

1. `music-genre-identifier-CI` debe terminar correctamente.
2. `music-genre-identifier-CD` debe comenzar después.
3. El último step debe confirmar `/health`.

Un pull request ejecuta CI, pero no despliega. Un fallo de lint, tests o build
impide que se ejecute el job de deploy.

## Verificar el servicio

Obtener la URL:

```bash
gcloud run services describe music-genre-identifier \
  --project=gen-lang-client-0620673955 \
  --region=southamerica-west1 \
  --format='value(status.url)'
```

Comprobar proceso y modelo:

```bash
curl https://music-genre-identifier-j7ngpo2fqq-tl.a.run.app/health
curl https://music-genre-identifier-j7ngpo2fqq-tl.a.run.app/ready
```

Ver las revisiones:

```bash
gcloud run revisions list \
  --service=music-genre-identifier \
  --project=gen-lang-client-0620673955 \
  --region=southamerica-west1
```

Leer logs recientes:

```bash
gcloud run services logs read music-genre-identifier \
  --project=gen-lang-client-0620673955 \
  --region=southamerica-west1 \
  --limit=50
```

## Pausar o desactivar

“Desactivar” puede significar cosas diferentes. Cloud Run no tiene un botón de
apagado equivalente al de una máquina virtual.

### Evitar nuevos deploys automáticos

Esto no baja la API actual. Solo pausa el CD.

En GitHub:

```text
Actions -> music-genre-identifier-CD -> menú (...) -> Disable workflow
```

Con GitHub CLI:

```bash
gh workflow disable deploy.yml
```

Para reactivarlo:

```bash
gh workflow enable deploy.yml
```

### Mantener el servicio pero quitar el acceso público

Esta es la opción reversible si no se quiere que nadie invoque la API:

```bash
gcloud run services remove-iam-policy-binding music-genre-identifier \
  --project=gen-lang-client-0620673955 \
  --region=southamerica-west1 \
  --member='allUsers' \
  --role='roles/run.invoker'
```

La URL seguirá existiendo, pero las peticiones anónimas recibirán un error de
autorización. El health check del CD también fallará mientras el servicio sea
privado.

Para volver a hacerlo público:

```bash
gcloud run services add-iam-policy-binding music-genre-identifier \
  --project=gen-lang-client-0620673955 \
  --region=southamerica-west1 \
  --member='allUsers' \
  --role='roles/run.invoker'
```

### Reducir el uso en reposo

El servicio ya tiene `min=0`. No hay que eliminarlo para que escale a cero:

```bash
gcloud run services update music-genre-identifier \
  --project=gen-lang-client-0620673955 \
  --region=southamerica-west1 \
  --min=0 \
  --max=1
```

Un request puede volver a crear una instancia. Si se necesita impedir todo uso,
hay que quitar el acceso público o eliminar el servicio.

### Eliminar el servicio

Esta operación elimina el servicio y sus revisiones:

```bash
gcloud run services delete music-genre-identifier \
  --project=gen-lang-client-0620673955 \
  --region=southamerica-west1
```

No debe hacerse para un apagado temporal. Si el CD vuelve a crear el servicio,
el recurso nuevo puede quedar privado. Será necesario restaurar el binding de
`allUsers`, y el health check fallará hasta hacerlo.

Eliminar Cloud Run tampoco elimina automáticamente las imágenes de Artifact
Registry.

## Cambiar CPU, memoria o escalado

Modificar directamente el servicio:

```bash
gcloud run services update music-genre-identifier \
  --project=gen-lang-client-0620673955 \
  --region=southamerica-west1 \
  --cpu=1 \
  --memory=2Gi \
  --concurrency=1 \
  --min=0 \
  --max=1
```

El workflow actual no declara estos valores. Al desplegar una imagen nueva,
Cloud Run conserva la configuración del servicio existente. Si se quiere que
el repositorio sea la fuente de verdad, se pueden añadir al step
`deploy-cloudrun` mediante su input `flags`:

```yaml
flags: >-
  --cpu=1
  --memory=2Gi
  --concurrency=1
  --min=0
  --max=1
```

Antes de aumentar memoria, CPU o máximo de instancias hay que revisar el efecto
en costos. `max=1` limita el crecimiento, pero también limita la capacidad para
atender requests simultáneos.

## Imágenes y política de limpieza

La imagen actual ocupa aproximadamente 220 MB comprimidos. El Dockerfile copia
solo `app/`, `model/` y las dependencias de producción. No incluye el dataset ni
el código de entrenamiento.

El archivo `artifact-cleanup-policy.json` define dos reglas:

1. Todas las versiones son candidatas a eliminación.
2. La versión más reciente siempre se conserva.

La regla `KEEP` tiene prioridad sobre `DELETE`. La política está activa, no en
dry run. Artifact Registry la ejecuta de forma asíncrona, así que después de un
deploy pueden aparecer varias versiones durante un tiempo.

Editar el JSON no cambia Google Cloud por sí solo. Para probar una modificación:

```bash
gcloud artifacts repositories set-cleanup-policies music-genre-identifier \
  --project=gen-lang-client-0620673955 \
  --location=southamerica-west1 \
  --policy=artifact-cleanup-policy.json \
  --dry-run
```

Para activarla:

```bash
gcloud artifacts repositories set-cleanup-policies music-genre-identifier \
  --project=gen-lang-client-0620673955 \
  --location=southamerica-west1 \
  --policy=artifact-cleanup-policy.json \
  --no-dry-run
```

Esta política minimiza almacenamiento, pero sacrifica historial en Artifact
Registry. Una revisión anterior de Cloud Run puede seguir disponible durante
un tiempo porque Cloud Run importa la imagen al desplegarla, pero no existe un
rollback durable a todas las imágenes anteriores.

## Rollback

Listar revisiones:

```bash
gcloud run revisions list \
  --service=music-genre-identifier \
  --project=gen-lang-client-0620673955 \
  --region=southamerica-west1
```

Enviar todo el tráfico a una revisión que todavía exista:

```bash
gcloud run services update-traffic music-genre-identifier \
  --project=gen-lang-client-0620673955 \
  --region=southamerica-west1 \
  --to-revisions=REVISION_NAME=100
```

Si esa revisión ya no existe y su imagen fue eliminada por la política, será
necesario volver a construir el commit correspondiente. Por eso los tags usan
el SHA de Git y no `latest`.

## Deploy manual de emergencia

El CD es la vía normal. Estos pasos sirven si GitHub Actions no está disponible.

```bash
gcloud auth configure-docker southamerica-west1-docker.pkg.dev

docker build \
  -t southamerica-west1-docker.pkg.dev/gen-lang-client-0620673955/music-genre-identifier/music-genre-identifier:manual .

docker push \
  southamerica-west1-docker.pkg.dev/gen-lang-client-0620673955/music-genre-identifier/music-genre-identifier:manual

gcloud run deploy music-genre-identifier \
  --project=gen-lang-client-0620673955 \
  --region=southamerica-west1 \
  --image=southamerica-west1-docker.pkg.dev/gen-lang-client-0620673955/music-genre-identifier/music-genre-identifier:manual
```

Después hay que comprobar `/health` y `/ready`. Un deploy manual no reemplaza
la necesidad de arreglar el workflow si el CD está fallando.

## Qué se configuró una sola vez

Estos componentes ya existen y no se recrean en cada deploy:

- APIs de Cloud Run, Artifact Registry, IAM, IAM Credentials y STS.
- Repositorio Docker de Artifact Registry.
- Servicio de Cloud Run.
- Cuenta `github-deployer`.
- Workload Identity Pool y proveedor OIDC.
- Bindings IAM descritos anteriormente.
- Variables del repositorio de GitHub.
- Política de limpieza de Artifact Registry.

El proveedor de identidad es:

```text
projects/845382797070/locations/global/workloadIdentityPools/github-actions/providers/github
```

Su issuer es:

```text
https://token.actions.githubusercontent.com/
```

La relación completa es:

```text
GitHub Actions
    -> token OIDC firmado por GitHub
    -> proveedor github del pool github-actions
    -> principal del repositorio GaryVidalC/music-genre-identifier
    -> service account github-deployer
    -> permisos limitados sobre Artifact Registry y Cloud Run
```

## Errores comunes

### CI termina bien pero CD no comienza

Revisar:

- que el workflow de CI se llame exactamente `music-genre-identifier-CI`;
- que el evento haya sido un push a `main`, no un pull request;
- que `deploy.yml` esté habilitado en GitHub Actions;
- que el commit exista en la rama remota.

### Falla la autenticación

Revisar `id-token: write`, las variables `GCP_WORKLOAD_IDENTITY_PROVIDER` y
`GCP_SERVICE_ACCOUNT`, y la condición del proveedor OIDC. No crear una clave
JSON como solución rápida.

### Docker no puede hacer push

Comprobar que el URI contiene región, proyecto, repositorio, imagen y tag, en
ese orden. Verificar también `roles/artifactregistry.writer` sobre el
repositorio correcto.

### Cloud Run rechaza el deploy

Comprobar `roles/run.developer` sobre el servicio y
`roles/iam.serviceAccountUser` sobre la cuenta de ejecución. Si se cambió el
nombre del servicio, los permisos anteriores no se trasladan automáticamente.

### El health check devuelve 403

El servicio dejó de ser público. Revisar que `allUsers` tenga
`roles/run.invoker`.

### El health check devuelve 500 o no conecta

Revisar los logs de Cloud Run. `/health` solo confirma que el proceso responde;
`/ready` comprueba además que el modelo y sus metadatos estén cargados.

### La imagen nueva no aparece o se despliega otra versión

El valor de `image` en el step de Cloud Run debe ser idéntico al valor de
`tags` usado por Buildx. Ambos deben terminar en el SHA obtenido desde
`steps.checkout.outputs.ref`.

## Control de costos

Las decisiones actuales para limitar consumo son:

- cero instancias mínimas;
- máximo de una instancia;
- una CPU y 2 GiB de memoria;
- CPU asignada durante requests;
- una sola versión conservada en Artifact Registry.

Esto reduce el riesgo, pero no garantiza costo cero. Pueden existir cargos por
requests, tiempo de cómputo, transferencia de datos, builds y almacenamiento.
Los presupuestos de Google Cloud sirven como alerta, no como corte automático
del servicio.

Antes de una prueba de carga o de aumentar `max`, CPU o memoria, revisar la
facturación y los límites configurados.

## Notas para agentes

Antes de modificar el despliegue:

1. Leer `.github/workflows/ci.yml`, `.github/workflows/deploy.yml`, `Dockerfile`
   y `artifact-cleanup-policy.json`.
2. Tratar el estado de Google Cloud como estado externo que puede diferir del
   repositorio. Usar comandos `describe` antes de asumir valores.
3. Mantener la relación entre el tag construido y la imagen desplegada.
4. Mantener el checkout de `github.event.workflow_run.head_sha` para desplegar
   exactamente el commit aprobado por CI.
5. No permitir deploys desde pull requests. Un workflow con `workflow_run`
   tiene permisos sensibles y no debe ejecutar código no confiable.
6. Mantener Workload Identity Federation. No añadir claves JSON de cuentas de
   servicio al repositorio ni a GitHub.
7. No eliminar servicios, imágenes, repositorios o bindings IAM sin una
   autorización explícita.
8. Recordar que modificar el archivo de cleanup no aplica la política. Requiere
   ejecutar `gcloud artifacts repositories set-cleanup-policies`.
9. Después de un cambio, comprobar CI, CD, `/health`, `/ready` y los logs.

La configuración de recursos de Cloud Run todavía vive principalmente en el
servicio remoto. El workflow fija la imagen, el nombre, el proyecto y la región,
pero no CPU, memoria, concurrencia ni escalado. Si se busca una infraestructura
completamente reproducible, esos valores deben declararse en el workflow o en
una herramienta de infraestructura como código.

## Referencias oficiales

- [Deploy to Cloud Run GitHub Action](https://github.com/google-github-actions/deploy-cloudrun)
- [Eventos `workflow_run` de GitHub Actions](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)
- [OIDC en GitHub Actions](https://docs.github.com/en/actions/reference/security/oidc)
- [Administrar servicios de Cloud Run](https://docs.cloud.google.com/run/docs/managing/services)
- [Acceso público y privado en Cloud Run](https://docs.cloud.google.com/run/docs/securing/managing-access)
- [Políticas de limpieza de Artifact Registry](https://docs.cloud.google.com/artifact-registry/docs/repositories/cleanup-policy)
