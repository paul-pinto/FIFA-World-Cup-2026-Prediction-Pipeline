param(
    [string]$EvalDate = "",
    [string]$PredictDate = "",
    [switch]$FetchOdds = $true,
    [switch]$Telegram = $false
)

$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\DESCARGAS\PYPAUL\worldcup_model"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LogDir = Join-Path $ProjectRoot "logs"

if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

if ($PredictDate -eq "") {
    $PredictDate = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd")
}

if ($EvalDate -eq "") {
    $EvalDate = (Get-Date).ToUniversalTime().AddDays(-1).ToString("yyyy-MM-dd")
}

$Timestamp = (Get-Date).ToString("yyyyMMdd_HHmmss")
$LogFile = Join-Path $LogDir "pipeline_$Timestamp.log"

Set-Location $ProjectRoot

Write-Host "============================================================"
Write-Host "World Cup Predictor Daily Pipeline"
Write-Host "============================================================"
Write-Host "ProjectRoot : $ProjectRoot"
Write-Host "Python      : $Python"
Write-Host "EvalDate    : $EvalDate"
Write-Host "PredictDate : $PredictDate"
Write-Host "FetchOdds   : $FetchOdds"
Write-Host "LogFile     : $LogFile"
Write-Host "Telegram    : $Telegram"
Write-Host "============================================================"

if (!(Test-Path $Python)) {
    throw "Python no encontrado en $Python"
}

$argsList = @(
    "-m", "src.pipeline",
    "full",
    "--eval-date", $EvalDate,
    "--predict-date", $PredictDate
)

if ($FetchOdds) {
    $argsList += "--fetch-odds"
}

if ($Telegram) {
    $argsList += "--telegram"
}

& $Python @argsList *>&1 | Tee-Object -FilePath $LogFile

if ($LASTEXITCODE -ne 0) {
    Write-Host "Pipeline failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "============================================================"
Write-Host "Pipeline completed OK"
Write-Host "============================================================"