# Delete_filetype_helper.ps1
# Called from Delete_filetype.bat
# Env vars: ROOT, EXTS_FOR_PS (semicolon-separated), MODE ("scan" or "recycle")

Add-Type -AssemblyName Microsoft.VisualBasic

$root        = $env:ROOT
$exts        = $env:EXTS_FOR_PS -split ';' | Where-Object { $_ -ne '' }
$mode        = $env:DFT_MODE   # "scan" or "recycle"

$totalFound    = 0
$totalRecycled = 0

foreach ($ext in $exts) {
    $pattern = '*.' + $ext
    $files   = @(Get-ChildItem -LiteralPath $root -Filter $pattern -File -Recurse -Force -ErrorAction SilentlyContinue)
    $found   = $files.Count
    $totalFound += $found

    if ($mode -eq 'recycle') {
        $recycled = 0
        foreach ($f in $files) {
            try {
                if ($f.Attributes -match 'ReadOnly') {
                    $f.Attributes = $f.Attributes -band -not [System.IO.FileAttributes]::ReadOnly
                }
                [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile(
                    $f.FullName, 'OnlyErrorDialogs', 'SendToRecycleBin'
                )
                if (-not (Test-Path -LiteralPath $f.FullName)) { $recycled++ }
            } catch {}
        }
        $totalRecycled += $recycled
        if ($found -gt 0) {
            Write-Host ('  .' + $ext.PadRight(8) + 'recycled ' + $recycled + ' / ' + $found)
        }
    } else {
        # scan mode
        Write-Host ('  .' + $ext.PadRight(8) + $found.ToString().PadLeft(5) + ' file(s)')
    }
}

Write-Host ''
if ($mode -eq 'recycle') {
    Write-Host ('Done. Total recycled: ' + $totalRecycled + ' / ' + $totalFound + ' file(s).')
} else {
    Write-Host ('Total: ' + $totalFound + ' file(s) to be recycled.')
}
