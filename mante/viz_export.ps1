# Game-page data for another checkpoint (same exports as the w_base page): run in mante\viz_NAME, which holds the
# checkpoint as count_w_w_base.npz (the name the trough, cross-section and 3-D exports read).  CPU only.
#   powershell -File viz_export.ps1 -Name p_transfer
param([string]$Name)
$d = Join-Path $PSScriptRoot "viz_$Name"; Set-Location $d
$env:CUDA_VISIBLE_DEVICES = ''
$log = "export.log"; "start $(Get-Date -Format 'HH:mm:ss') $Name" | Out-File $log -Encoding utf8
$data = (Resolve-Path "..\..\rnn\out_data.json").Path
$steps = @(
  @('..\count_pack.py', 'count_w_w_base.npz', 'count_b64.txt'),
  @('..\count_frames2.py', $data, 'count_w_w_base.npz', 'count_frames_w.json'),
  @('..\count_landscape_export.py', 'count_w_w_base.npz', 'count_land.json'),
  @('..\count_rib3d_export.py'),
  @('..\count_trough_export.py'),
  @('..\count_cross_export.py'))
foreach ($s in $steps) {
  $t = Get-Date; $out = & python @s 2>&1 | Out-String
  "$(Get-Date -Format 'HH:mm:ss') exit $LASTEXITCODE ($([int]((Get-Date) - $t).TotalSeconds) s): $($s -join ' ')`n$($out.Trim())" | Out-File $log -Append -Encoding utf8
  if ($LASTEXITCODE -ne 0) { "FAILED" | Out-File $log -Append -Encoding utf8; exit 1 }
}
"done $(Get-Date -Format 'HH:mm:ss')" | Out-File $log -Append -Encoding utf8
