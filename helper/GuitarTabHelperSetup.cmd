@echo off
title Guitar Tab link helper setup
echo Setting up the Guitar Tab link helper (yt-dlp + Deno, about 60 MB)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol=3072; iex ((New-Object Net.WebClient).DownloadString('https://guitar-tab.onrender.com/helper/install.ps1'))"
echo.
pause
