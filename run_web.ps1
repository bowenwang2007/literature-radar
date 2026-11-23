$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$App = Join-Path $ProjectRoot "src\eduecon_radar\ui\app.py"
$EnvFile = Join-Path $ProjectRoot ".env"
$Port = 8501

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    Write-Error "未找到项目虚拟环境 Python：$Python"
    exit 1
}

if (-not (Test-Path -LiteralPath $App -PathType Leaf)) {
    Write-Error "未找到 Streamlit 入口：$App"
    exit 1
}

$Listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($Listeners) {
    $OwnerPid = @($Listeners | ForEach-Object { $_.OwningProcess } | Sort-Object -Unique)[0]
    Write-Host "端口 $Port 已被占用，现有进程 PID=$OwnerPid。"
    Write-Host "请先关闭旧进程："
    Write-Host "Stop-Process -Id $OwnerPid -Force"
    Write-Host "Literature Radar 不允许两个实例同时运行。"
    exit 1
}

$OwnerPids = @()
$NetstatLines = netstat -ano | Select-String ":$Port\s" | Select-String "LISTENING"
foreach ($Line in $NetstatLines) {
    $Parts = ($Line.ToString().Trim() -split "\s+")
    if ($Parts.Count -ge 5 -and $Parts[$Parts.Count - 1] -match "^\d+$") {
        $OwnerPids += [int]$Parts[$Parts.Count - 1]
    }
}
if ($OwnerPids) {
    $OwnerPid = @($OwnerPids | Sort-Object -Unique)[0]
    Write-Host "端口 $Port 已被占用，现有进程 PID=$OwnerPid。"
    Write-Host "请先关闭旧进程："
    Write-Host "Stop-Process -Id $OwnerPid -Force"
    Write-Host "Literature Radar 不允许两个实例同时运行。"
    exit 1
}

if (Test-Path -LiteralPath $EnvFile -PathType Leaf) {
    foreach ($Line in Get-Content -LiteralPath $EnvFile -Encoding UTF8) {
        $Trimmed = $Line.Trim()
        if (-not $Trimmed -or $Trimmed.StartsWith("#") -or -not $Trimmed.Contains("=")) {
            continue
        }
        $Parts = $Trimmed.Split("=", 2)
        $Key = $Parts[0].Trim()
        $Value = $Parts[1].Trim().Trim('"').Trim("'")
        if ($Key -and $null -eq [Environment]::GetEnvironmentVariable($Key, "Process")) {
            [Environment]::SetEnvironmentVariable($Key, $Value, "Process")
        }
    }
}

$env:LITERATURE_RADAR_MODE = "local"
Set-Location -LiteralPath $ProjectRoot
Write-Host "Literature Radar Local App"
Write-Host "Python: $Python"
Write-Host "Entry: $App"
Write-Host "URL: http://localhost:8501/"
& $Python -m streamlit run $App --server.address localhost --server.port $Port
