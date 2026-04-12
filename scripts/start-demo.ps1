param(
    [string]$ApiHost = "127.0.0.1",
    [int]$ApiPort = 8000,
    [int]$UiPort = 8501
)

$ErrorActionPreference = "Stop"

Write-Host "Starting demo stack from project root..."

if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    throw "Virtual environment not found at .\\venv\\Scripts\\python.exe"
}

# Start FastAPI server in a new terminal window
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$PWD'; .\\venv\\Scripts\\python.exe -m uvicorn app.main:app --host $ApiHost --port $ApiPort --reload"
)

# Start Streamlit UI in another new terminal window
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$PWD'; .\\venv\\Scripts\\python.exe -m streamlit run app/ui.py --server.port $UiPort"
)

Write-Host "API expected at:  http://$ApiHost`:$ApiPort/docs"
Write-Host "UI expected at:   http://$ApiHost`:$UiPort"
Write-Host "Tip: wait a few seconds, then start screen recording."
