param(
    [ValidateRange(1, 1000)][int]$MaxRounds = 30,
    [ValidateRange(1, 20)][int]$NoProgressLimit = 3
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

# One named mutex per repository prevents two Codex sessions from modifying the
# same worktree.  The hash keeps the Windows object name short and stable.
$sha256 = [System.Security.Cryptography.SHA256]::Create()
try {
    $scopeBytes = $sha256.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($ProjectRoot.ToLowerInvariant()))
    $scope = ([System.BitConverter]::ToString($scopeBytes)).Replace('-', '').Substring(0, 20)
} finally { $sha256.Dispose() }
$createdNew = $false
$autopilotMutex = [System.Threading.Mutex]::new($true, "Local\BuildingAIAutopilot_$scope", [ref]$createdNew)
if (-not $createdNew) {
    $autopilotMutex.Dispose()
    Write-Host 'AUTOPILOT ALREADY RUNNING'
    exit 0
}

try {
    $required = @('AGENTIC_PLATFORM_ROADMAP.md', 'scripts/run_agentic_acceptance.py', 'scripts/CODEX_AUTOPILOT_PROMPT.md')
    foreach ($path in $required) { if (-not (Test-Path $path)) { throw "Required file missing: $path" } }
    $autoRoot = Join-Path $ProjectRoot 'artifacts/autopilot'
    New-Item -ItemType Directory -Force -Path $autoRoot | Out-Null
    $sessionId = ('{0:yyyyMMdd_HHmmss}_pid_{1}' -f (Get-Date), $PID)

    function Invoke-Acceptance {
        & python scripts/run_agentic_acceptance.py | Out-Host
        $jsonPath = Join-Path $ProjectRoot 'artifacts/acceptance/latest.json'
        if (-not (Test-Path $jsonPath)) { throw 'Acceptance did not create latest.json.' }
        return (Get-Content $jsonPath -Raw -Encoding UTF8 | ConvertFrom-Json)
    }
    function Git-Commit { (git rev-parse --short HEAD).Trim() }
    function Invoke-CodexAttempt([int]$Round, [int]$Attempt) {
        $prefix = Join-Path $autoRoot ('run_{0}_round_{1:D3}_attempt_{2:D2}' -f $sessionId, $Round, $Attempt)
        $stdoutLog = "$prefix.stdout.log"
        $stderrLog = "$prefix.stderr.log"
        # Each stream has its own new file.  The synchronous pipeline waits for
        # codex to exit and closes Tee-Object/redirection handles before return.
        $priorPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            Get-Content 'scripts/CODEX_AUTOPILOT_PROMPT.md' -Raw -Encoding UTF8 |
                & $script:CodexCommand exec -C $ProjectRoot --sandbox workspace-write - 2> $stderrLog |
                Tee-Object -FilePath $stdoutLog | Out-Host
            return $LASTEXITCODE
        } finally { $ErrorActionPreference = $priorPreference }
    }

    $state = Invoke-Acceptance
    $noProgress = 0
    $started = Get-Date
    if ($state.counts.FAIL -gt 0) {
        $command = Get-Command codex -ErrorAction SilentlyContinue
        if (-not $command) { throw 'Codex CLI not found on PATH.' }
        $script:CodexCommand = $command.Source
        & $script:CodexCommand --version | Out-Host
    }
    for ($round = 1; $round -le $MaxRounds; $round++) {
        if ($state.counts.FAIL -eq 0) { break }
        Write-Host "`nBuildingAI Autopilot | Round $round / $MaxRounds | $(Git-Commit) | PASS $($state.counts.PASS) / FAIL $($state.counts.FAIL) / BLOCKED $($state.counts.ENVIRONMENT_BLOCKED)"
        $ok = $false
        for ($attempt = 1; $attempt -le 2; $attempt++) {
            $code = Invoke-CodexAttempt $round $attempt
            if ($code -eq 0) { $ok = $true; break }
            Start-Sleep -Seconds 3
        }
        if (-not $ok) { Write-Host "CODEX EXEC FAILED`nSession: $sessionId"; exit 2 }
        $before = $state
        $state = Invoke-Acceptance
        $passDelta = [int]$state.counts.PASS - [int]$before.counts.PASS
        $failDelta = [int]$state.counts.FAIL - [int]$before.counts.FAIL
        Write-Host "Round $round summary: Before PASS $($before.counts.PASS) / FAIL $($before.counts.FAIL); After PASS $($state.counts.PASS) / FAIL $($state.counts.FAIL); Commit $(Git-Commit)"
        if ($passDelta -le 0 -and $failDelta -ge 0) { $noProgress++ } else { $noProgress = 0 }
        if ($state.counts.FAIL -gt $before.counts.FAIL -and $round -gt 1) { Write-Host 'REGRESSION DETECTED'; exit 3 }
        if ($noProgress -ge $NoProgressLimit) { Write-Host "AUTOPILOT STALLED`nSession: $sessionId"; exit 4 }
    }
    if ($state.counts.FAIL -eq 0) {
        $pytest = & python -m pytest 2>&1
        $pytest | Out-Host
        $state = Invoke-Acceptance
        $report = @("session: $sessionId", "final commit: $(Git-Commit)", "pytest: $($pytest[-1])", "PASS: $($state.counts.PASS)", "FAIL: $($state.counts.FAIL)", "ENVIRONMENT_BLOCKED: $($state.counts.ENVIRONMENT_BLOCKED)", "elapsed: $((Get-Date)-$started)")
        $report | Set-Content (Join-Path $autoRoot "FINAL_REPORT_$sessionId.txt") -Encoding UTF8
        Write-Host '================================'
        Write-Host ($(if ($state.counts.ENVIRONMENT_BLOCKED -gt 0) { 'COMPLETE WITH ENVIRONMENT BLOCKERS' } else { 'BUILDINGAI AUTOPILOT COMPLETE' }))
        Write-Host '================================'
    } else {
        Write-Host "AUTOPILOT MAX ROUNDS REACHED | PASS $($state.counts.PASS) / FAIL $($state.counts.FAIL)"
        exit 5
    }
} finally {
    $autopilotMutex.ReleaseMutex()
    $autopilotMutex.Dispose()
}
