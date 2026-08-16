while ($true) {

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "STARTING PPO TRAINING"
    Write-Host "============================================================"

    python -m SCB_RL.trainer

    $exitCode = $LASTEXITCODE

    Write-Host ""
    Write-Host "Training process exited with code: $exitCode"

    if ($exitCode -eq 0) {

        Write-Host "Training completed successfully."
        break

    }

    Write-Host "Process crashed/interrupted."
    Write-Host "Restarting from latest checkpoint in 5 seconds..."

    Start-Sleep -Seconds 5
}