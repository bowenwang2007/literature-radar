$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$App = Join-Path $ProjectRoot "src\eduecon_radar\ui\app.py"
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
$PortBusy = [bool]$Listeners
if (-not $PortBusy) {
    $PortBusy = [bool](netstat -ano | Select-String ":$Port\s" | Select-String "LISTENING")
}
if ($PortBusy) {
    Write-Host "端口 $Port 已在使用。如果已启动 Literature Radar，请在浏览器打开 http://localhost:8501/ 继续阅读。"
    Write-Host "若该地址属于其他程序，请先正常关闭占用此端口的程序。"
    exit 1
}

$PreviousMode = $env:LITERATURE_RADAR_MODE
Push-Location -LiteralPath $ProjectRoot
try {
    $env:LITERATURE_RADAR_MODE = "local"
    Write-Host "Literature Radar Local App"
    Write-Host "Python: $Python"
    Write-Host "Entry: $App"
    Write-Host "URL: http://localhost:8501/"
    & $Python -m streamlit run $App --server.address localhost --server.port $Port
    $Result = $LASTEXITCODE
} finally {
    $env:LITERATURE_RADAR_MODE = $PreviousMode
    Pop-Location
}
exit $Result
