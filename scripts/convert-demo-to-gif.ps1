param(
    [Parameter(Mandatory = $true)]
    [string]$InputMp4,
    [string]$OutputGif = "docs/assets/demo.gif",
    [int]$Fps = 12,
    [int]$Width = 960
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $InputMp4)) {
    throw "Input file not found: $InputMp4"
}

$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    throw "ffmpeg is required. Install ffmpeg and retry."
}

$outputDir = Split-Path -Parent $OutputGif
if ($outputDir -and -not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

$palettePath = Join-Path $env:TEMP "demo_palette.png"

Write-Host "Generating palette..."
$paletteFilter = ("fps={0},scale={1}:-1:flags=lanczos,palettegen" -f $Fps, $Width)
& ffmpeg -y -i $InputMp4 -vf $paletteFilter $palettePath | Out-Null

Write-Host "Encoding GIF..."
$gifFilter = ("fps={0},scale={1}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5" -f $Fps, $Width)
& ffmpeg -y -i $InputMp4 -i $palettePath -lavfi $gifFilter $OutputGif | Out-Null

if (Test-Path $palettePath) {
    Remove-Item $palettePath -Force -ErrorAction SilentlyContinue
}

Write-Host "GIF created at: $OutputGif"
Write-Host "Tip: replace README demo placeholder path with this file if needed."
