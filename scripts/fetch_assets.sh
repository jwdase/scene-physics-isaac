#!/usr/bin/env bash
# Fetch the render assets that are too large for git (CC0): table wood + floor tile textures
# (ambientCG) and the interior HDRI (Poly Haven). Run from anywhere:
#     bash scripts/fetch_assets.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEX="$ROOT/assets/textures"
HDRI="$ROOT/assets/hdri"
UA="Mozilla/5.0 (X11; Linux x86_64)"
mkdir -p "$TEX" "$HDRI"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fetch_acg() {  # $1 = ambientCG asset id
  local id="$1"
  echo "ambientCG: $id (4K-JPG)"
  curl -fsSL -A "$UA" -o "$TMP/$id.zip" "https://ambientcg.com/get?file=${id}_4K-JPG.zip"
  rm -rf "$TEX/$id" && mkdir -p "$TEX/$id"
  unzip -oq "$TMP/$id.zip" -d "$TEX/$id"
  find "$TEX/$id" -type f ! \( -iname "*_Color.jpg" -o -iname "*_Roughness.jpg" -o -iname "*_Displacement.jpg" \) -delete
}

fetch_hdri() {  # $1 = Poly Haven slug, $2 = resolution
  echo "PolyHaven: ${1}_${2}.hdr"
  curl -fsSL -A "$UA" -o "$HDRI/${1}_${2}.hdr" "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/${2}/${1}_${2}.hdr"
}

fetch_acg Wood063        # table top
fetch_acg Tiles108       # floor
fetch_hdri lythwood_room 4k
fetch_hdri studio_small_08 2k
echo "Done: $TEX, $HDRI"
