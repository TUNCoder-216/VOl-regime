# Manual deploy: same steps the webhook runs. Usage (repo root): .\deploy\deploy.ps1
$ErrorActionPreference = "Stop"
docker compose pull
docker compose up -d --force-recreate --remove-orphans
docker compose logs model-fetcher --no-color --tail 15
docker compose ps
