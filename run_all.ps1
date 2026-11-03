$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python train.py --data data --out outputs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python postprocess.py --data data --prob outputs\res34_thinned.tif --out outputs\fault_map.tif
