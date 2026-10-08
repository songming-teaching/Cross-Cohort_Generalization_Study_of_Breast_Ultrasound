# Detached launch of TIER4 training.
# Why: the harness kills background commands at a 2-hour limit, but the
# remaining TIER4 runs need ~10.7 h (16 runs x ~40 min). Two prior attempts
# were killed by that limit. Start-Process detaches from this shell's job
# object, so the training survives.
# Note: do NOT name a variable $args -- it is a PowerShell automatic variable.

$root = "C:\Users\15315\Desktop\paper1st\meliorate\project"
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "src"

$py  = "$root\..\.venv\Scripts\python.exe"
$log = "$root\runs\_logs\tier4_detached.log"
$err = "$root\runs\_logs\tier4_detached.err"

$argList = @(
  "-u", "-m", "breastdg.train.dev_screen",
  "--queue", "tier4",
  "--directions", "busbra,busi",
  "--seeds", "42,123,2026",
  "--steps", "3000",
  "--eval-every", "500"
)

$p = Start-Process -FilePath $py -ArgumentList $argList -NoNewWindow -PassThru `
     -RedirectStandardOutput $log -RedirectStandardError $err

Write-Output "detached pid=$($p.Id)"
Write-Output "log=$log"
