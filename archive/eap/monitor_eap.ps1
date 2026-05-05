$baseline = 'logs\eap_baseline_qk15_it120_out.log'
$candidate = 'logs\eap_qkgain5_it120_out.log'

Write-Host 'Monitoring EAP validation runs...'
Write-Host 'Baseline - QK_GAIN=1.5 (expected winner)'
Write-Host 'Candidate - QK_GAIN=5.0 (expected loser)'
Write-Host ''

for ($i = 0; $i -lt 15; $i++) {
    Start-Sleep -Seconds 60
    $b_ok = (Test-Path $baseline) -and ((Get-Content $baseline -Tail 5 -ErrorAction SilentlyContinue) -match 'Export complete')
    $c_ok = (Test-Path $candidate) -and ((Get-Content $candidate -Tail 5 -ErrorAction SilentlyContinue) -match 'Export complete')
    $ts = Get-Date -Format 'HH:mm:ss'
    Write-Host "$ts | Baseline $(if($b_ok){'DONE'}else{'running ...'}) | Candidate $(if($c_ok){'DONE'}else{'running ...'})"
    if ($b_ok -and $c_ok) {
        Write-Host ''
        Write-Host '=== Both complete! Running EAP comparison ==='
        Write-Host ''
        & .\venv\Scripts\python eap_fit.py $baseline
        Write-Host ''
        & .\venv\Scripts\python eap_fit.py $candidate
        Write-Host ''
        Write-Host '=== Comparing L values ==='
        $b_match = (Get-Content $baseline -Tail 30 | Select-String 'L  = ([\d.]+)')
        $c_match = (Get-Content $candidate -Tail 30 | Select-String 'L  = ([\d.]+)')
        break
    }
}