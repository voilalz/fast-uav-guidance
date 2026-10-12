#!/usr/bin/env bash
# 下载视觉管线用的公开数据到 <data_dir>（不随仓库分发），之后运行 vision_build_map 生成地图文件。
#
#   tools/download_data.sh data            # ahr（默认，约 150 MB）
#   tools/download_data.sh data/la la      # Louisiana（约 1 GB）
#
# 区域：
#   ahr  SpaceNet 8 Germany（CC BY-SA 4.0），德国 Ahr 河谷，网格第 28–29 行、第 59–68 列，
#        0.84 km × 5.4 km 条带，森林山地 + 城镇；Copernicus GLO-30 DEM N50 E006/E007
#   la   SpaceNet 8 Louisiana-East（CC BY-SA 4.0），新奥尔良以东，第 1 组瓦片第 4–14 行、第 72–105 列，
#        约 5.5 km × 14.6 km（有空洞），平原城郊 + 湿地 + 水体；Copernicus GLO-30 DEM N29 W090/W091
set -euo pipefail
D=${1:-data}
AREA=${2:-ahr}
S3=https://spacenet-dataset.s3.amazonaws.com/spacenet/SN8_floods
DEM=https://copernicus-dem-30m.s3.amazonaws.com
case "$AREA" in
  ahr) SET=Germany_Training_Public; GROUP=""; R0=28; R1=29; C0=59; C1=68
       DEMS="N50_00_E006_00 N50_00_E007_00" ;;
  la)  SET=Louisiana-East_Training_Public; GROUP=1; R0=4; R1=14; C0=72; C1=105
       DEMS="N29_00_W091_00 N29_00_W090_00" ;;
  *) echo "未知区域 $AREA（ahr | la）" >&2; exit 1 ;;
esac
mkdir -p "$D/sn8/pre" "$D/sn8/post" "$D/dem"

get() {  # url path
  [ -s "$2" ] || curl -fsSL --retry 4 -o "$2" "$1"
}

get "$S3/$SET/${SET}_label_image_mapping.csv" "$D/sn8/mapping.csv"
# 列：label, pre-event image, post-event image 1, ...；label 形如 <组>_<行>_<列>.geojson
tail -n +2 "$D/sn8/mapping.csv" | tr -d '\r' | while IFS=, read -r label pre post _; do
  rc=${label%.geojson}
  col=${rc##*_}
  rest=${rc%_*}
  row=${rest##*_}
  grp=${rest%_*}
  [ -z "$GROUP" ] || [ "$grp" = "$GROUP" ] || continue
  if [ "$row" -ge $R0 ] && [ "$row" -le $R1 ] && [ "$col" -ge $C0 ] && [ "$col" -le $C1 ]; then
    get "$S3/$SET/PRE-event/$pre" "$D/sn8/pre/$pre"
    get "$S3/$SET/POST-event/$post" "$D/sn8/post/$post"
  fi
done
for t in $DEMS; do
  n=Copernicus_DSM_COG_10_${t}_DEM
  get "$DEM/$n/$n.tif" "$D/dem/cop30_${t}.tif"
done
echo "下载完成：$D"
