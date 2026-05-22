# Deploy Student Adaptability ML Dashboard to GitHub and Streamlit Cloud
# Run from PowerShell in the project folder after GitHub CLI login succeeds.

$ErrorActionPreference = "Stop"
$env:Path = "C:\Program Files\Git\cmd;C:\Program Files\GitHub CLI;" + $env:Path

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

$RepoName = "student-adaptability-ml-dashboard"

Write-Host "Checking GitHub authentication..."
gh auth status
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Not logged in. Starting device login..."
    Write-Host "1. Copy the one-time code shown below"
    Write-Host "2. Open https://github.com/login/device"
    Write-Host "3. Paste the code and authorize GitHub CLI"
    gh auth login --hostname github.com --git-protocol https --device
}

Write-Host ""
Write-Host "Creating GitHub repository and pushing..."
gh repo create $RepoName --public --source=. --remote=origin --push --description "Streamlit dashboard for student adaptability ML prediction"

$remoteUrl = gh repo view --json url -q .url
Write-Host ""
Write-Host "Repository published: $remoteUrl"
Write-Host ""
Write-Host "Next: deploy on Streamlit Cloud"
Write-Host "  1. Open https://share.streamlit.io"
Write-Host "  2. Sign in with GitHub"
Write-Host "  3. New app -> select $RepoName, branch main, main file streamlit_app.py"
Write-Host ""
