param(
    [Parameter(Mandatory=$true)]
    [string]$DeploymentZip
)
$ErrorActionPreference = "Stop"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "Git for Windows belum terpasang." }
if (-not (Get-Command git-lfs -ErrorAction SilentlyContinue)) { throw "Git LFS belum terpasang. Install dari https://git-lfs.com/." }
if (-not (Test-Path ".git")) { throw "Jalankan dari folder hasil git clone repository VOISS." }
if (-not (Test-Path $DeploymentZip)) { throw "ZIP tidak ditemukan: $DeploymentZip" }

$temp = Join-Path $env:TEMP ("voiss_models_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $temp | Out-Null
try {
    Expand-Archive -LiteralPath $DeploymentZip -DestinationPath $temp -Force
    $names = @(
        "SeismicNet_final.keras",
        "VOISSNet_final.keras",
        "SeismicNet_VOISSNet_Multimodal_final.keras"
    )
    New-Item -ItemType Directory -Force -Path "artifacts" | Out-Null
    foreach ($n in $names) {
        $src = Join-Path $temp $n
        if (-not (Test-Path $src)) { throw "File model tidak ditemukan dalam ZIP: $n" }
        Copy-Item -LiteralPath $src -Destination (Join-Path "artifacts" $n) -Force
    }
    git lfs install
    git lfs track "artifacts/*.keras"
    git add .gitattributes
    git add "artifacts/SeismicNet_final.keras" "artifacts/VOISSNet_final.keras" "artifacts/SeismicNet_VOISSNet_Multimodal_final.keras"
    Write-Host "Model yang terdeteksi Git LFS:" -ForegroundColor Cyan
    git lfs ls-files
    foreach ($n in $names) {
        if (-not (git lfs ls-files | Select-String -SimpleMatch $n)) { throw "File belum terlacak LFS: $n" }
    }
    git commit -m "Add trained VOISS model artifacts"
    if ($LASTEXITCODE -ne 0) { throw "Commit gagal. Jika tidak ada perubahan, periksa apakah model sudah ditambahkan sebelumnya." }
    git push origin main
    if ($LASTEXITCODE -ne 0) { throw "Push gagal. Periksa autentikasi GitHub dan kuota Git LFS." }
    Write-Host "Berhasil: model VOISS di-push ke GitHub." -ForegroundColor Green
}
finally { Remove-Item -LiteralPath $temp -Recurse -Force -ErrorAction SilentlyContinue }
