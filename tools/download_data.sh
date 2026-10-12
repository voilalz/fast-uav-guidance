#!/usr/bin/env bash
# 下载视觉管线用的公开数据到 <data_dir>（默认 data/，约 150 MB，不随仓库分发）：
#   * SpaceNet 8 Germany（CC BY-SA 4.0）：Ahr 河谷网格第 28–29 行、第 59–68 列的 PRE/POST 影像瓦片
#   * Copernicus GLO-30 DEM：N50 E006、E007 两个 1° 瓦片
# 之后运行 vision_build_map <data_dir> 生成地图文件。
set -euo pipefail
D=${1:-data}
SN8=https://spacenet-dataset.s3.amazonaws.com/spacenet/SN8_floods/Germany_Training_Public
DEM=https://copernicus-dem-30m.s3.amazonaws.com
mkdir -p "$D/sn8/pre" "$D/sn8/post" "$D/dem"

get() {  # url path
  [ -s "$2" ] || curl -fsSL --retry 4 -o "$2" "$1"
}

get "$SN8/Germany_Training_Public_label_image_mapping.csv" "$D/sn8/mapping.csv"
# 列：label, pre-event image, post-event image 1, ...；label 形如 <id>_<row>_<col>.geojson
tail -n +2 "$D/sn8/mapping.csv" | tr -d '\r' | while IFS=, read -r label pre post _; do
  rc=${label%.geojson}
  col=${rc##*_}
  rest=${rc%_*}
  row=${rest##*_}
  if [ "$row" -ge 28 ] && [ "$row" -le 29 ] && [ "$col" -ge 59 ] && [ "$col" -le 68 ]; then
    get "$SN8/PRE-event/$pre" "$D/sn8/pre/$pre"
    get "$SN8/POST-event/$post" "$D/sn8/post/$post"
  fi
done
for e in 6 7; do
  n=Copernicus_DSM_COG_10_N50_00_E00${e}_00_DEM
  get "$DEM/$n/$n.tif" "$D/dem/cop30_N50_E00${e}.tif"
done
echo "下载完成：$D"
