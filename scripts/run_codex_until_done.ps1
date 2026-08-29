param(
    [ValidateRange(1, 1000)][int]$MaxRounds = 30,
    [ValidateRange(1, 20)][int]$NoProgressLimit = 3
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot
$required = @('AGENTIC_PLATFORM_ROADMAP.md','scripts/run_agentic_acceptance.py','scripts/CODEX_AUTOPILOT_PROMPT.md')
foreach ($path in $required) { if (-not (Test-Path $path)) { throw "Required file missing: $path" } }
if (-not (Get-Command codex -ErrorAction SilentlyContinue)) { throw 'Codex CLI not found on PATH.' }
& codex --version | Out-Host
$autoRoot = Join-Path $ProjectRoot 'artifacts/autopilot'; New-Item -ItemType Directory -Force -Path $autoRoot | Out-Null
function Invoke-Acceptance {
    & python scripts/run_agentic_acceptance.py | Out-Host
    $jsonPath = Join-Path $ProjectRoot 'artifacts/acceptance/latest.json'
    if (-not (Test-Path $jsonPath)) { throw 'Acceptance did not create latest.json.' }
    return (Get-Content $jsonPath -Raw -Encoding UTF8 | ConvertFrom-Json)
}
function Git-Commit { (git rev-parse --short HEAD).Trim() }
$state = Invoke-Acceptance; $noProgress = 0; $started = Get-Date
for ($round=1; $round -le $MaxRounds; $round++) {
    if ($state.counts.FAIL -eq 0) { break }
    Write-Host "`nBuildingAI Autopilot | Round $round / $MaxRounds | $(Git-Commit) | PASS $($state.counts.PASS) / FAIL $($state.counts.FAIL) / BLOCKED $($state.counts.ENVIRONMENT_BLOCKED)"
    $log = Join-Path $autoRoot ('run_{0:D3}.log' -f $round)
    $ok = $false
    for ($attempt=1; $attempt -le 2; $attempt++) {
        # Use stdin explicitly: multiline Markdown is otherwise fragile as a
        # Windows native-command argument and Codex supports '-' for this mode.
        $priorPreference=$ErrorActionPreference; $ErrorActionPreference='Continue'
        Get-Content 'scripts/CODEX_AUTOPILOT_PROMPT.md' -Raw -Encoding UTF8 | & codex exec -C $ProjectRoot --sandbox workspace-write - *>&1 | Tee-Object -FilePath $log -Append | Out-Host
        $code=$LASTEXITCODE; $ErrorActionPreference=$priorPreference
        if ($code -eq 0) { $ok = $true; break }
        Start-Sleep -Seconds 3
    }
    if (-not $ok) { Write-Host "CODEX EXEC FAILED`nLog: $log"; exit 2 }
    $before=$state; $state=Invoke-Acceptance
    $passDelta=[int]$state.counts.PASS-[int]$before.counts.PASS; $failDelta=[int]$state.counts.FAIL-[int]$before.counts.FAIL
    Write-Host "Round $round summary: Before PASS $($before.counts.PASS) / FAIL $($before.counts.FAIL); After PASS $($state.counts.PASS) / FAIL $($state.counts.FAIL); Commit $(Git-Commit)"
    if ($passDelta -le 0 -and $failDelta -ge 0) { $noProgress++ } else { $noProgress=0 }
    if ($state.counts.FAIL -gt $before.counts.FAIL -and $round -gt 1) { Write-Host 'REGRESSION DETECTED'; exit 3 }
    if ($noProgress -ge $NoProgressLimit) { Write-Host "AUTOPILOT STALLED`nLast logs: $log"; exit 4 }
}
if ($state.counts.FAIL -eq 0) {
    $pytest=& python -m pytest 2>&1; $pytest | Out-Host; $state=Invoke-Acceptance
    $report=@("final commit: $(Git-Commit)","pytest: $($pytest[-1])","PASS: $($state.counts.PASS)","FAIL: $($state.counts.FAIL)","ENVIRONMENT_BLOCKED: $($state.counts.ENVIRONMENT_BLOCKED)","elapsed: $((Get-Date)-$started)")
    $report | Set-Content (Join-Path $autoRoot 'FINAL_REPORT.txt') -Encoding UTF8
    Write-Host '================================'; Write-Host ($(if ($state.counts.ENVIRONMENT_BLOCKED -gt 0) {'COMPLETE WITH ENVIRONMENT BLOCKERS'} else {'BUILDINGAI AUTOPILOT COMPLETE'})); Write-Host '================================'
} else { Write-Host "AUTOPILOT MAX ROUNDS REACHED | PASS $($state.counts.PASS) / FAIL $($state.counts.FAIL)"; exit 5 }
