# Launch the TIER4 resume driver in its OWN console (ASCII only).
# Do NOT add -NoNewWindow: see the comment block in _tier4_driver.ps1.

$root   = "C:\Users\15315\Desktop\paper1st\meliorate\project"
$driver = "$root\scripts\_tier4_driver.ps1"

$p = Start-Process -FilePath "powershell" `
     -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $driver) `
     -WindowStyle Hidden -PassThru

"$($p.Id)" | Set-Content -Path "$root\runs\_logs\tier4_driver.pid" -Encoding utf8
Write-Output "driver pid=$($p.Id)"
