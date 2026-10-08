# TIER4 auto-resume driver (ASCII only -- PowerShell reads .ps1 as GBK on this
# machine, so any non-ASCII byte in this file breaks parsing).
#
# Why this exists: the detached run died silently at 2026-10-05 05:31 (GPU idle
# for 10 h). dev_screen's own skip logic is keyed on results.csv and is
# idempotent, so an outer "restart if dead" loop makes interruption a non-issue.
#
# Launched by _launch_tier4_driver.ps1 in its OWN console (no -NoNewWindow):
# -NoNewWindow binds the child to the parent console, so destroying that
# console delivers CTRL_CLOSE_EVENT and kills the child. That is the most
# plausible cause of the earlier death, so it is avoided explicitly here.

$ErrorActionPreference = "Continue"

$root = "C:\Users\15315\Desktop\paper1st\meliorate\project"
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH       = "src"

$py      = "$root\..\.venv\Scripts\python.exe"
$DriverLog = "$root\runs\_logs\tier4_driver.log"
$runlog    = "$root\runs\_logs\tier4_detached.log"

function Note($m) {
  "$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))  $m" |
    Add-Content -Path $DriverLog -Encoding utf8
}

Note "driver start pid=$PID"
"=== driver start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" |
  Add-Content -Path $runlog -Encoding utf8

for ($attempt = 1; $attempt -le 40; $attempt++) {
  Note "attempt $attempt : launching dev_screen tier4"
  "--- attempt $attempt  $(Get-Date -Format 'HH:mm:ss') ---" |
    Add-Content -Path $runlog -Encoding utf8

  & $py -u -m breastdg.train.dev_screen `
      --queue tier4 --directions busbra,busi `
      --seeds 42,123,2026 --steps 3000 --eval-every 500 2>&1 |
    Tee-Object -FilePath $runlog -Append

  $code = $LASTEXITCODE
  Note "attempt $attempt exited code=$code"
  if ($code -eq 0) { Note "clean exit, driver stops"; break }
  Start-Sleep -Seconds 15
}

Note "driver end"
