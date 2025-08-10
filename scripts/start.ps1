# 设置控制台编码为 UTF-8，确保中文正常显示
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'

<#
SemanticCoreDB Windows 启动脚本 (PowerShell)

用法示例：
  - 启动服务:           powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1
  - 开发模式:           powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1 --dev
  - Docker 启动:        powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1 --docker
  - 安装依赖:           powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1 --install
  - 运行测试:           powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1 --test
  - 运行示例:           powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1 --example
  - 查看帮助:           powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1 --help
#>

function Write-Info($msg)   { Write-Host "[INFO]  $msg" -ForegroundColor Green }
function Write-Warn($msg)   { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Write-Err($msg)    { Write-Host "[ERROR] $msg" -ForegroundColor Red }

function Get-PythonCmd {
  $candidates = @(
    @{ cmd = 'py'; args = '-3' },
    @{ cmd = 'python'; args = '' },
    @{ cmd = 'python3'; args = '' }
  )
  foreach ($c in $candidates) {
    $exists = Get-Command $c.cmd -ErrorAction SilentlyContinue
    if ($exists) {
      if ($c.args) { return "$($c.cmd) $($c.args)" } else { return $c.cmd }
    }
  }
  return $null
}

function Check-PythonVersion {
  Write-Info "检查 Python 版本..."
  $python = Get-PythonCmd
  if (-not $python) { Write-Err "未找到 Python，请先安装 Python 3.9+"; exit 1 }
  $verOut = & $python --version 2>&1
  Write-Info "检测到: $verOut"
  if ($verOut -match '(\d+)\.(\d+)') {
    $major = [int]$Matches[1]; $minor = [int]$Matches[2]
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 9)) {
      Write-Err "需要 Python 3.9+，当前: $verOut"; exit 1
    }
  }
  return $python
}

function Check-Dependencies($python) {
  Write-Info "检查依赖..."
  $pipOk = $false
  try { & $python -m pip --version *> $null; $pipOk = $true } catch {}
  if (-not $pipOk) { Write-Err "未检测到 pip，请先安装 pip"; exit 1 }
  if (-not (Test-Path -Path 'requirements.txt')) {
    Write-Warn "未找到 requirements.txt，跳过依赖安装"
  }
}

function Install-Dependencies($python) {
  if (Test-Path -Path 'requirements.txt') {
    Write-Info "安装 Python 依赖..."
    & $python -m pip install --upgrade pip
    & $python -m pip install -r requirements.txt
  }
}

function Create-Directories {
  Write-Info "创建必要目录..."
  $dirs = @(
    'data', 'data/events', 'data/objects', 'data/vectors', 'data/metadata',
    'logs', 'temp'
  )
  foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
}

function Check-Environment {
  Write-Info "检查环境变量..."
  if (-not $env:OPENAI_API_KEY) { Write-Warn "OPENAI_API_KEY 未设置，部分功能可能不可用" }
  if (-not $env:SCDB_HOST) { $env:SCDB_HOST = 'localhost' }
  if (-not $env:SCDB_PORT) { $env:SCDB_PORT = '8000' }
}

function Start-Database($python) {
  Write-Info "使用本地 Python 启动 API..."
  & $python -m src.api.main
}

function Start-DatabaseDocker {
  Write-Info "使用 Docker 启动..."
  $dc = Get-Command docker-compose -ErrorAction SilentlyContinue
  if (-not $dc) { Write-Err "未检测到 docker-compose，请先安装 Docker Desktop"; exit 1 }
  docker-compose up -d
}

function Start-Dev($python) {
  Write-Info "开发模式启动 (热重载)..."
  if (-not $env:SCDB_PORT) { $env:SCDB_PORT = '8000' }
  & $python -m uvicorn src.api.main:app --host 0.0.0.0 --port $env:SCDB_PORT --reload
}

function Run-Tests($python) {
  Write-Info "运行测试..."
  & $python -m pytest tests -v
}

function Run-Examples($python) {
  Write-Info "运行示例..."
  if (Test-Path 'examples/basic_usage.py') {
    & $python examples/basic_usage.py
  } else {
    Write-Warn "未找到 examples/basic_usage.py"
  }
}

function Show-Help {
  Write-Host "SemanticCoreDB Windows 启动脚本" -ForegroundColor Cyan
  Write-Host ""
  Write-Host "用法: powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1 [选项]"
  Write-Host ""
  Write-Host "选项:" 
  Write-Host "  --help | -h       显示此帮助"
  Write-Host "  --install         安装依赖并创建目录"
  Write-Host "  --test            运行测试"
  Write-Host "  --example         运行示例"
  Write-Host "  --docker          使用 Docker 启动"
  Write-Host "  --dev             开发模式(热重载)"
  Write-Host "  (默认无参数)      本地启动 API"
}

# 主流程
$action = if ($args -and $args.Count -gt 0) { $args[0] } else { '' }
$python = Check-PythonVersion
Check-Dependencies $python
Create-Directories
Check-Environment

switch ($action) {
  '--help' { Show-Help; break }
  '-h'     { Show-Help; break }
  '--install' { Install-Dependencies $python; Write-Info '安装完成'; break }
  '--test'    { Run-Tests $python; break }
  '--example' { Run-Examples $python; break }
  '--docker'  { Start-DatabaseDocker; break }
  '--dev'     { Start-Dev $python; break }
  default     { Start-Database $python; break }
}