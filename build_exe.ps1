[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$buildWork = Join-Path ([System.IO.Path]::GetTempPath()) "sensorlab-pyinstaller"

Push-Location -LiteralPath $projectRoot
try {
    py -3 -m PyInstaller `
        --noconfirm `
        --clean `
        --onefile `
        --windowed `
        --name SensorLab `
        --hidden-import matplotlib.backends.backend_pdf `
        --workpath $buildWork `
        --paths "$projectRoot\src" `
        "$projectRoot\sensorlab_gui.py"
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

Write-Host "Built $projectRoot\dist\SensorLab.exe"
