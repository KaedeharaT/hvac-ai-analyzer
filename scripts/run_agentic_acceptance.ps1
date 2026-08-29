$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
python scripts/run_agentic_acceptance.py
exit $LASTEXITCODE
