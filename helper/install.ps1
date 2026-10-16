# Installs the Guitar Tab link helper for the current Windows user (no admin rights needed):
#   %LOCALAPPDATA%\GuitarTabHelper\helper.ps1   the helper (see that file for what it does)
#   %LOCALAPPDATA%\GuitarTabHelper\yt-dlp.exe   downloads the audio (github.com/yt-dlp/yt-dlp)
#   %LOCALAPPDATA%\GuitarTabHelper\deno.exe     JavaScript runtime yt-dlp needs for YouTube (deno.com)
# plus "Guitar Tab helper" shortcuts on the desktop and in the Start menu, then starts it.
# Run it again any time to update. Uninstall: delete that folder and the two shortcuts.
#
# One line for PowerShell:
#   [Net.ServicePointManager]::SecurityProtocol=3072; iex ((New-Object Net.WebClient).DownloadString('https://guitar-tab.onrender.com/helper/install.ps1'))

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue' # downloads are far faster without Windows PowerShell's progress bar
try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch { }

$site = 'https://guitar-tab.onrender.com'
$dir = Join-Path $env:LOCALAPPDATA 'GuitarTabHelper'
New-Item -ItemType Directory -Force -Path $dir | Out-Null
Write-Host ''
Write-Host "Installing the Guitar Tab link helper in $dir" -ForegroundColor Cyan

# stop a running copy so its files can be replaced
Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe' OR Name = 'pwsh.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -like '*GuitarTabHelper*helper.ps1*' -and $_.ProcessId -ne $PID } |
  ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force } catch { } }

Write-Host '  helper...'
Invoke-WebRequest -UseBasicParsing -Uri "$site/helper/helper.ps1" -OutFile (Join-Path $dir 'helper.ps1')
Write-Host '  yt-dlp (18 MB)...'
Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe' -OutFile (Join-Path $dir 'yt-dlp.exe')
if (-not (Test-Path (Join-Path $dir 'deno.exe'))) {
  Write-Host '  Deno, which yt-dlp needs for YouTube (43 MB)...'
  $zip = Join-Path $env:TEMP 'guitar-tab-deno.zip'
  Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/denoland/deno/releases/latest/download/deno-x86_64-pc-windows-msvc.zip' -OutFile $zip
  Expand-Archive -Force -Path $zip -DestinationPath $dir
  Remove-Item $zip -Force
}

$ps = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
$helperArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $dir 'helper.ps1')`""
$shell = New-Object -ComObject WScript.Shell
$places = @([Environment]::GetFolderPath('Desktop'), (Join-Path ([Environment]::GetFolderPath('Programs')) ''))
foreach ($place in $places) {
  if (-not $place -or -not (Test-Path $place)) { continue }
  $lnk = $shell.CreateShortcut((Join-Path $place 'Guitar Tab helper.lnk'))
  $lnk.TargetPath = $ps
  $lnk.Arguments = $helperArgs
  $lnk.WorkingDirectory = $dir
  $lnk.WindowStyle = 7 # minimized
  $lnk.IconLocation = "$env:WINDIR\System32\imageres.dll,103"
  $lnk.Description = 'Lets Guitar Tab take songs straight from links'
  $lnk.Save()
}

Start-Process -FilePath $ps -ArgumentList $helperArgs -WorkingDirectory $dir -WindowStyle Minimized
Write-Host ''
Write-Host 'Done. The helper is running in a minimized window.' -ForegroundColor Green
Write-Host 'Go back to Guitar Tab and paste a link. If your browser asks to let the site'
Write-Host 'reach apps on this device, choose Allow (that is how it talks to the helper).'
Write-Host 'Next time, start it from the "Guitar Tab helper" shortcut on your desktop.'
Write-Host ''
