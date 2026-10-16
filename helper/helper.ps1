# Guitar Tab link helper
#
# Lets the Guitar Tab app (https://guitar-tab.onrender.com) take a song straight from a link:
# YouTube, YouTube Music, SoundCloud, Bandcamp, Vimeo and most other sites yt-dlp knows, plus
# Spotify / Apple Music / Deezer / Tidal links (looked up on YouTube by artist and title, since
# those services encrypt their audio). The download runs here, on your own computer and
# internet connection, with yt-dlp (https://github.com/yt-dlp/yt-dlp), and the audio goes
# straight to the app's page. Nothing is uploaded anywhere.
#
# It only listens on 127.0.0.1 (this computer), only answers the Guitar Tab page, and keeps
# just the last song (in %LOCALAPPDATA%\GuitarTabHelper\jobs). Close this window to stop it.
# Use it for music you're allowed to download.
#
# Works with Windows PowerShell 5.1 (built into Windows 10/11) and PowerShell 7.
param([int]$Port = 47823)

$ErrorActionPreference = 'Stop'
try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch { }

$HelperVersion = '1.0'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$OnWindows = ($PSVersionTable.PSEdition -ne 'Core') -or $IsWindows
$YtDlp = Join-Path $Here $(if ($OnWindows) { 'yt-dlp.exe' } else { 'yt-dlp' })
$Deno = Join-Path $Here $(if ($OnWindows) { 'deno.exe' } else { 'deno' })
$Work = Join-Path $Here 'jobs'
New-Item -ItemType Directory -Force -Path $Work | Out-Null
$AllowedOrigins = @('https://guitar-tab.onrender.com')
$LocalOrigin = '^https?://(localhost|127\.0\.0\.1)(:\d+)?$'
$UserAgent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36'

function Say([string]$text) { Write-Host ('[' + (Get-Date -Format 'HH:mm:ss') + '] ' + $text) }

if (-not (Test-Path $YtDlp)) {
  Say 'Downloading yt-dlp (once)...'
  $name = $(if ($OnWindows) { 'yt-dlp.exe' } else { 'yt-dlp_linux' })
  Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/yt-dlp/yt-dlp/releases/latest/download/$name" -OutFile $YtDlp
  if (-not $OnWindows) { & chmod +x $YtDlp }
}
# YouTube changes often: keep yt-dlp current
try { Say ('yt-dlp: ' + ((& $YtDlp -U 2>&1 | Select-Object -Last 1) -as [string])) } catch { Say 'yt-dlp: could not check for updates' }
$YtDlpVersion = ((& $YtDlp --version 2>$null) -as [string]).Trim()

# ---------------------------------------------------------------- jobs (one song at a time)
$script:Job = $null

function Q([string]$s) { '"' + ($s -replace '"', '\"') + '"' }

function Get-PageText([string]$url) {
  (Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 20 -Headers @{ 'User-Agent' = $UserAgent; 'Accept-Language' = 'en-US,en;q=0.8' }).Content
}

function Get-Meta([string]$html, [string]$prop) {
  $m = [regex]::Match($html, '<meta[^>]+(?:property|name)="' + [regex]::Escape($prop) + '"[^>]+content="([^"]*)"', 'IgnoreCase')
  if (-not $m.Success) { $m = [regex]::Match($html, '<meta[^>]+content="([^"]*)"[^>]+(?:property|name)="' + [regex]::Escape($prop) + '"', 'IgnoreCase') }
  if ($m.Success) { return [System.Net.WebUtility]::HtmlDecode($m.Groups[1].Value).Trim() }
  return ''
}

# Streaming services encrypt their audio: find the same recording on YouTube by artist and title.
function Resolve-Target([string]$url) {
  $uri = [Uri]$url
  $hostName = $uri.Host.ToLower()
  if ($hostName -notmatch '(^|\.)(open\.spotify\.com|spotify\.link|music\.apple\.com|deezer\.com|deezer\.page\.link|link\.deezer\.com|tidal\.com|listen\.tidal\.com|music\.amazon\.[a-z.]+)$') { return @{ target = $url; label = '' } }
  $title = ''
  $artist = ''
  $dot = [char]0x00B7
  # Spotify tracks: the embed page carries the track's name and artists as data
  $m = [regex]::Match($url, 'open\.spotify\.com/(?:intl-[a-z]+/)?track/([A-Za-z0-9]+)')
  if ($m.Success) {
    try {
      $emb = Get-PageText "https://open.spotify.com/embed/track/$($m.Groups[1].Value)"
      $nd = [regex]::Match($emb, '<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', 'Singleline')
      if ($nd.Success) {
        $e = ($nd.Groups[1].Value | ConvertFrom-Json).props.pageProps.state.data.entity
        if ($e.name) { $title = [string]$e.name; if ($e.artists) { $artist = [string]$e.artists[0].name } }
      }
    } catch { }
  }
  if (-not $title) {
    # link previews: these pages give social-media crawlers the song's title and artist
    $html = (Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 20 -Headers @{ 'User-Agent' = 'facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)'; 'Accept-Language' = 'en-US,en;q=0.8' }).Content
    $og = Get-Meta $html 'og:title'
    $desc = Get-Meta $html 'og:description'
    $pageTitle = [System.Net.WebUtility]::HtmlDecode(([regex]::Match($html, '<title[^>]*>([^<]*)</title>', 'IgnoreCase')).Groups[1].Value).Trim()
    $m = [regex]::Match($pageTitle, '^(.+?) - (?:Song|Single|Album)[^|]*? by (.+?) \| Spotify$', 'IgnoreCase')
    if ($m.Success) { $title = $m.Groups[1].Value; $artist = $m.Groups[2].Value }
    else {
      $m = [regex]::Match($og, '^(.+?) by (.+?)(?: on (?:Apple Music|TIDAL|Deezer|Amazon Music))?$', 'IgnoreCase')
      if ($m.Success) { $title = $m.Groups[1].Value; $artist = $m.Groups[2].Value }
      elseif ($og) {
        $title = $og
        if ($desc.Contains([string]$dot)) { $artist = $desc.Split($dot)[0].Trim() }
      }
    }
  }
  $title = ($title -replace '^\W+', '' -replace '[\u200E\u200F"]', '').Trim()
  $artist = ($artist -replace '[\u200E\u200F"]', '').Trim()
  if (-not $title) { throw "Couldn't read the song's name from that page." }
  $q = $(if ($artist) { "$artist - $title" } else { $title })
  return @{ target = "ytsearch1:$q official audio"; label = $q }
}

function Start-Download([string]$url) {
  if ($script:Job -and $script:Job.proc -and -not $script:Job.proc.HasExited) { try { $script:Job.proc.Kill() } catch { } }
  $id = [Guid]::NewGuid().ToString('N').Substring(0, 12)
  Get-ChildItem -Path $Work -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
  $dir = Join-Path $Work $id
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  $r = Resolve-Target $url
  $out = Join-Path $dir 'audio.%(ext)s'
  # progressive (plain HTTP) audio first: every browser can decode those files as they are
  $ytArgs = @('--no-playlist', '--no-warnings', '--newline', '--no-part', '--no-mtime', '--no-colors',
    '-f', 'bestaudio[ext=m4a][protocol^=http]/bestaudio[protocol^=http]/bestaudio[ext=m4a]/bestaudio/best',
    '--max-filesize', '500M', '--write-info-json', '-o', (Q $out))
  if (Test-Path $Deno) { $ytArgs += @('--js-runtimes', (Q ('deno:' + $Deno))) }
  $ytArgs += @('--', (Q $r.target))
  $log = Join-Path $dir 'log.txt'
  $err = Join-Path $dir 'err.txt'
  $p = Start-Process -FilePath $YtDlp -ArgumentList $ytArgs -RedirectStandardOutput $log -RedirectStandardError $err -NoNewWindow -PassThru
  $null = $p.Handle # keeps the exit code readable later (Windows PowerShell quirk)
  $script:Job = @{ id = $id; dir = $dir; proc = $p; log = $log; err = $err; url = $url; label = $r.label; started = Get-Date }
  Say ("Getting " + $(if ($r.label) { "$($r.label) (from YouTube)" } else { $url }))
  return $id
}

function Get-AudioFile($job) {
  Get-ChildItem -Path $job.dir -File | Where-Object { $_.Name -like 'audio.*' -and $_.Name -notlike '*.info.json' -and $_.Name -notlike '*.part' -and $_.Name -notlike '*.ytdl' } | Select-Object -First 1
}

function Get-JobStatus($job) {
  $s = [ordered]@{ id = $job.id; state = 'running'; progress = 0; title = $job.label; error = '' }
  if (Test-Path $job.log) {
    $text = ''
    try { $fs = [IO.File]::Open($job.log, 'Open', 'Read', 'ReadWrite'); $sr = New-Object IO.StreamReader($fs); $text = $sr.ReadToEnd(); $sr.Close() } catch { }
    $ms = [regex]::Matches($text, '\[download\]\s+([\d.]+)%')
    if ($ms.Count -gt 0) { $s.progress = [double]::Parse($ms[$ms.Count - 1].Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture) / 100 }
  }
  $info = Get-ChildItem -Path $job.dir -Filter '*.info.json' -File -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($info) {
    try { $j = Get-Content -Raw -Encoding UTF8 $info.FullName | ConvertFrom-Json; if ($j.title) { $s.title = $(if ($j.artist -and $j.track) { "$($j.artist) - $($j.track)" } else { $j.title }) } } catch { }
  }
  if ($job.proc.HasExited) {
    $f = Get-AudioFile $job
    if ($job.proc.ExitCode -eq 0 -and $f) { $s.state = 'done'; $s.progress = 1 }
    else {
      $s.state = 'error'
      $e = ''
      if (Test-Path $job.err) { $e = ((Get-Content $job.err -ErrorAction SilentlyContinue) | Where-Object { $_ -match 'ERROR' } | Select-Object -Last 1) -as [string] }
      if (-not $e) { $e = 'The download did not finish.' }
      $s.error = ($e -replace '^ERROR:\s*', '').Trim()
    }
  }
  return $s
}

# ---------------------------------------------------------------- tiny HTTP server
function Send-Response($ns, [int]$code, [hashtable]$headers, [byte[]]$body, [string]$file) {
  $reason = @{ 200 = 'OK'; 204 = 'No Content'; 400 = 'Bad Request'; 403 = 'Forbidden'; 404 = 'Not Found'; 409 = 'Conflict'; 500 = 'Internal Server Error' }[$code]
  $len = 0
  if ($file) { $len = (Get-Item $file).Length } elseif ($body) { $len = $body.Length }
  $sb = New-Object System.Text.StringBuilder
  [void]$sb.Append("HTTP/1.1 $code $reason`r`n")
  foreach ($k in $headers.Keys) { [void]$sb.Append("${k}: $($headers[$k])`r`n") }
  [void]$sb.Append("Content-Length: $len`r`nConnection: close`r`nCache-Control: no-store`r`n`r`n")
  $hb = [Text.Encoding]::ASCII.GetBytes($sb.ToString())
  $ns.Write($hb, 0, $hb.Length)
  if ($file) { $fs = [IO.File]::OpenRead($file); try { $fs.CopyTo($ns, 65536) } finally { $fs.Close() } }
  elseif ($body) { $ns.Write($body, 0, $body.Length) }
  $ns.Flush()
}

function Send-Json($ns, [int]$code, [hashtable]$cors, $obj) {
  $h = @{} + $cors
  $h['Content-Type'] = 'application/json; charset=utf-8'
  Send-Response $ns $code $h ([Text.Encoding]::UTF8.GetBytes(($obj | ConvertTo-Json -Compress))) $null
}

function Get-Query([string]$target) {
  $q = @{}
  $i = $target.IndexOf('?')
  if ($i -ge 0) {
    foreach ($pair in $target.Substring($i + 1).Split('&')) {
      $kv = $pair.Split(@('='), 2, [StringSplitOptions]::None)
      if ($kv.Length -eq 2) { $q[$kv[0]] = [Uri]::UnescapeDataString($kv[1].Replace('+', ' ')) }
    }
  }
  return $q
}

function Handle-Client($client) {
  $client.ReceiveTimeout = 5000
  $client.SendTimeout = 60000
  $ns = $client.GetStream()
  $buf = New-Object byte[] 16384
  $total = 0
  $text = ''
  while ($total -lt $buf.Length) {
    $n = $ns.Read($buf, $total, $buf.Length - $total)
    if ($n -le 0) { break }
    $total += $n
    $text = [Text.Encoding]::ASCII.GetString($buf, 0, $total)
    if ($text.Contains("`r`n`r`n")) { break }
  }
  if (-not $text) { return }
  $lines = ($text.Split(@("`r`n`r`n"), 2, [StringSplitOptions]::None)[0]) -split "`r`n"
  $first = $lines[0] -split ' '
  if ($first.Length -lt 2) { return }
  $method = $first[0].ToUpper()
  $target = $first[1]
  $h = @{}
  for ($i = 1; $i -lt $lines.Length; $i++) {
    $c = $lines[$i].IndexOf(':')
    if ($c -gt 0) { $h[$lines[$i].Substring(0, $c).Trim().ToLower()] = $lines[$i].Substring($c + 1).Trim() }
  }
  # only this computer, by IP or name (guards against DNS rebinding)
  $hostHeader = [string]$h['host']
  if ($hostHeader -ne "127.0.0.1:$Port" -and $hostHeader -ne "localhost:$Port") { Send-Response $ns 403 @{} $null $null; return }
  $origin = [string]$h['origin']
  $path = $target.Split('?')[0]
  if (-not $origin) {
    if ($method -eq 'GET' -and $path -eq '/') {
      $page = "<!doctype html><meta charset=utf-8><title>Guitar Tab link helper</title><body style='font:16px system-ui;margin:2em'><h1>Guitar Tab link helper is running</h1><p>Go back to <a href='https://guitar-tab.onrender.com/'>Guitar Tab</a> and paste a link. Close the helper's window to stop it.</p><p style='color:#666'>Helper $HelperVersion, yt-dlp $YtDlpVersion</p>"
      Send-Response $ns 200 @{ 'Content-Type' = 'text/html; charset=utf-8' } ([Text.Encoding]::UTF8.GetBytes($page)) $null
    } else { Send-Response $ns 403 @{} $null $null }
    return
  }
  if (-not ($AllowedOrigins -contains $origin -or $origin -match $LocalOrigin)) { Send-Response $ns 403 @{} $null $null; return }
  $cors = @{
    'Access-Control-Allow-Origin' = $origin
    'Vary' = 'Origin'
    'Access-Control-Allow-Methods' = 'GET, POST, OPTIONS'
    'Access-Control-Allow-Headers' = 'X-GuitarTab'
    'Access-Control-Allow-Private-Network' = 'true'
    'Access-Control-Expose-Headers' = 'X-Title, Content-Length'
    'Access-Control-Max-Age' = '600'
  }
  if ($method -eq 'OPTIONS') { Send-Response $ns 204 $cors $null $null; return }
  # a header simple cross-site requests can't carry: every call needs the preflight above
  if (-not $h.ContainsKey('x-guitartab')) { Send-Response $ns 403 $cors $null $null; return }
  $q = Get-Query $target
  switch ($path) {
    '/ping' { Send-Json $ns 200 $cors @{ ok = $true; helper = $HelperVersion; ytdlp = $YtDlpVersion; deno = (Test-Path $Deno) } }
    '/start' {
      $url = [string]$q['url']
      if ($url.Length -gt 2000 -or $url -notmatch '^https?://[^\s"<>`]+$') { Send-Json $ns 400 $cors @{ error = 'That is not a web link.' }; return }
      try { $id = Start-Download $url; Send-Json $ns 200 $cors @{ id = $id } }
      catch { Send-Json $ns 500 $cors @{ error = $_.Exception.Message } }
    }
    '/status' {
      if (-not $script:Job -or $script:Job.id -ne $q['id']) { Send-Json $ns 404 $cors @{ error = 'No such download.' }; return }
      Send-Json $ns 200 $cors (Get-JobStatus $script:Job)
    }
    '/audio' {
      if (-not $script:Job -or $script:Job.id -ne $q['id']) { Send-Json $ns 404 $cors @{ error = 'No such download.' }; return }
      $f = Get-AudioFile $script:Job
      if (-not $f -or -not $script:Job.proc.HasExited) { Send-Json $ns 409 $cors @{ error = 'Not finished yet.' }; return }
      $st = Get-JobStatus $script:Job
      $types = @{ '.m4a' = 'audio/mp4'; '.mp4' = 'video/mp4'; '.webm' = 'audio/webm'; '.weba' = 'audio/webm'; '.mp3' = 'audio/mpeg'; '.ogg' = 'audio/ogg'; '.opus' = 'audio/ogg'; '.wav' = 'audio/wav'; '.flac' = 'audio/flac'; '.aac' = 'audio/aac' }
      $type = $types[$f.Extension.ToLower()]
      if (-not $type) { $type = 'application/octet-stream' }
      $h2 = @{} + $cors
      $h2['Content-Type'] = $type
      $h2['X-Title'] = [Uri]::EscapeDataString([string]$st.title)
      Send-Response $ns 200 $h2 $null $f.FullName
      Say ('Sent ' + $(if ($st.title) { $st.title } else { $f.Name }) + ' to the page')
    }
    '/cancel' {
      if ($script:Job -and $script:Job.id -eq $q['id'] -and -not $script:Job.proc.HasExited) { try { $script:Job.proc.Kill() } catch { } }
      Send-Json $ns 200 $cors @{ ok = $true }
    }
    default { Send-Json $ns 404 $cors @{ error = 'Not found.' } }
  }
}

$listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $Port)
try { $listener.Start() } catch {
  Say "Port $Port is busy: the helper is probably already running. You can close this window."
  if ($OnWindows) { Start-Sleep -Seconds 8 }
  exit 1
}
Say "Guitar Tab link helper $HelperVersion is running (yt-dlp $YtDlpVersion$(if (Test-Path $Deno) { ', Deno ready for YouTube' } else { '' }))."
Say 'Leave this window open, go to https://guitar-tab.onrender.com/ and paste a link. Close this window to stop.'
try { $host.UI.RawUI.WindowTitle = 'Guitar Tab link helper' } catch { }
while ($true) {
  $client = $listener.AcceptTcpClient()
  try { Handle-Client $client } catch { Say ('Request failed: ' + $_.Exception.Message) } finally { try { $client.Close() } catch { } }
}
