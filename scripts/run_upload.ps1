<#
run_upload.ps1

Securely prompts for a GitHub Personal Access Token (PAT), installs PyGithub,
and runs the `upload_to_github_branch.py` script to upload the workspace to the
branch specified (default: Sathi_Nag).

Usage:
  Open PowerShell in the project root and run:
    .\scripts\run_upload.ps1 --owner Shuvam-Das --repo AlgoTradingWithZerodha --branch Sathi_Nag --path .

Notes:
- The script sets the token only for the current session. It attempts to clear
  sensitive values after use.
- Do NOT hard-code tokens in scripts.
#>

param(
    [string]$owner = "Shuvam-Das",
    [string]$repo = "AlgoTradingWithZerodha",
    [string]$branch = "Sathi_Nag",
    [string]$path = ".",
    [string]$base = "main"
)

function Write-Log {
    param([string]$msg)
    Write-Host "[run_upload] $msg"
}

# Prompt for token securely
Write-Log "You will be prompted to enter your GitHub Personal Access Token (repo scope)."
$secureToken = Read-Host -Prompt "Enter GitHub PAT (repo scope)" -AsSecureString
if (-not $secureToken) {
    Write-Error "No token provided. Exiting."
    exit 1
}

# Convert SecureString to plain for temporary use
$ptr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
$plainToken = [System.Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
[System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)

# Export token to environment for this session only
$env:GITHUB_TOKEN = $plainToken

try {
    # Ensure Python is available
    Write-Log "Checking Python availability..."
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        Write-Error "Python is not available in PATH. Please install Python 3.8+ and add it to PATH."
        exit 1
    }

    # Upgrade pip and install PyGithub
    Write-Log "Installing PyGithub..."
    python -m pip install --upgrade pip
    python -m pip install PyGithub

    # Run the upload script
    $scriptPath = Join-Path $PSScriptRoot "..\upload_to_github_branch.py"
    if (-not (Test-Path $scriptPath)) {
        # fallback: check project root
        $scriptPath = Join-Path (Resolve-Path ".") "upload_to_github_branch.py"
    }
    if (-not (Test-Path $scriptPath)) {
        Write-Error "upload_to_github_branch.py not found. Ensure script exists in repository root."
        exit 1
    }

    Write-Log "Running upload script to branch '$branch'..."
    & python $scriptPath --owner $owner --repo $repo --branch $branch --path $path --base $base

    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-Error "Upload script finished with exit code $exitCode"
        exit $exitCode
    }

    Write-Log "Upload script completed. GitHub Actions should be triggered automatically for branch '$branch'."
    Write-Log "Watch Actions in GitHub or wait and I can help inspect the run logs once it's started."
}
finally {
    # Clear sensitive values from environment and memory
    Write-Log "Clearing sensitive variables from session."
    Remove-Item Env:\GITHUB_TOKEN -ErrorAction SilentlyContinue
    $plainToken = $null
    $secureToken = $null
    [System.GC]::Collect()
}

Write-Log "Done."
