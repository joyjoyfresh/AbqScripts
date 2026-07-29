param(
    [string]$InputMarkdown = (Join-Path $PSScriptRoot '坡地地震动放大效应脚本通俗语音讲解.md'),
    [string]$OutputMp3 = (Join-Path $PSScriptRoot '坡地地震动放大效应脚本通俗语音讲解.mp3'),
    [string]$VoiceName = 'zh-CN-XiaoxiaoNeural',
    [string]$Rate = '-5%'
)

$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$markdown = Get-Content -LiteralPath $InputMarkdown -Raw -Encoding UTF8
$match = [regex]::Match(
    $markdown,
    '<!-- TTS-BEGIN -->([\s\S]*?)<!-- TTS-END -->',
    [System.Text.RegularExpressions.RegexOptions]::CultureInvariant
)
if (-not $match.Success) {
    throw '未找到 TTS-BEGIN/TTS-END 语音稿标记。'
}

$speechText = $match.Groups[1].Value.Trim()
$runtimeRoot = Join-Path ([IO.Path]::GetTempPath()) 'abqscripts-edge-tts-runtime'
$tempText = Join-Path ([IO.Path]::GetTempPath()) ('abqscripts-tts-' + [guid]::NewGuid().ToString('N') + '.txt')
$oldPythonPath = $env:PYTHONPATH
try {
    # 运行时只安装到系统临时目录，不修改项目依赖或 Python 环境。
    if (-not (Test-Path -LiteralPath (Join-Path $runtimeRoot 'edge_tts'))) {
        New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null
        & python -m pip install --disable-pip-version-check --quiet --target $runtimeRoot edge-tts
        if ($LASTEXITCODE -ne 0) {
            throw '临时安装 edge-tts 失败。'
        }
    }

    if ($oldPythonPath) {
        $env:PYTHONPATH = "$runtimeRoot$([IO.Path]::PathSeparator)$oldPythonPath"
    }
    else {
        $env:PYTHONPATH = $runtimeRoot
    }

    [IO.File]::WriteAllText($tempText, $speechText, [Text.UTF8Encoding]::new($false))
    & python -m edge_tts --file $tempText --voice $VoiceName "--rate=$Rate" --write-media $OutputMp3

    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $OutputMp3)) {
        throw 'edge-tts 未成功生成 MP3。'
    }
}
finally {
    if ($null -eq $oldPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONPATH = $oldPythonPath
    }
    if (Test-Path -LiteralPath $tempText) {
        Remove-Item -LiteralPath $tempText -Force
    }
}

Get-Item -LiteralPath $OutputMp3 | Select-Object FullName, Length, LastWriteTime
