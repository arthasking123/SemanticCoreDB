@echo off
setlocal

REM SemanticCoreDB Windows 启动包装器 (.bat)
REM 将所有参数透传给 PowerShell 脚本

set SCRIPT_DIR=%~dp0
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%start.ps1" %*

endlocal 