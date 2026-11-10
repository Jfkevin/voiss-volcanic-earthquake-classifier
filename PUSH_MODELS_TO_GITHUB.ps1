param(
    [Parameter(Mandatory=$true)]
    [string]$DeploymentZip
)
$ErrorActionPreference = "Stop"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "Git for Windows belum terpasang." }
if (-not (Get-Command git-lfs -ErrorAction SilentlyContinue)) { throw "Git LFS belum terpasang. Install dari https://git-lfs.com/ lalu buka PowerShell baru." }
if (-not (Test-Path ".git")) { throw "Jalankan script dari folder hasil git clone repository VOISS." }
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
    if ($LASTEXITCODE -ne 0) { throw "git lfs install gagal." }
    git lfs track "artifacts/*.keras"
    if ($LASTEXITCODE -ne 0) { throw "git lfs track gagal." }

    git add .gitattributes
    git add "artifacts/SeismicNet_final.keras" "artifacts/VOISSNet_final.keras" "artifacts/SeismicNet_VOISSNet_Multimodal_final.keras"

    Write-Host "Model yang terdaftar pada Git LFS:" -ForegroundColor Cyan
    git lfs ls-files
    foreach ($n in $names) {
        if (-not (git lfs ls-files | Select-String -SimpleMatch $n)) {
            throw "File belum terlacak LFS: $n"
        }
    }

    # Upload actual binary LFS objects referenced by main. Pointer files are already in the repository.
    git lfs push origin main
    if ($LASTEXITCODE -ne 0) { throw "Upload objek Git LFS gagal. Periksa autentikasi GitHub dan kuota LFS." }

    if (git status --porcelain) {
        git commit -m "Configure Git LFS for VOISS model artifacts"
        if ($LASTEXITCODE -ne 0) { throw "Commit konfigurasi LFS gagal." }
        git push origin main
        if ($LASTEXITCODE -ne 0) { throw "Push branch main gagal." }
    }

    Write-Host "Berhasil: tiga objek model telah diunggah ke Git LFS." -ForegroundColor Green
}
finally {
    Remove-Item -LiteralPath $temp -Recurse -Force -ErrorAction SilentlyContinue
}
