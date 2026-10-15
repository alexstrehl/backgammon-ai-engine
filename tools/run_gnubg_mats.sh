#!/usr/bin/env bash
# Usage: [D=dir] [MATS=subdir] [MET=table.xml] run_gnubg_mats.sh PLIES NPROC
# gnubg analysis of recorded matches: each D/MATS/match_*.mat is imported into
# its own gnubg process ("set threads 1"), NPROC of them in parallel, analysed
# PLIES deep (chequer play and cube decisions), and its "show statistics match"
# text written to D/out_p<PLIES>/<match>.txt; a match whose output is already
# complete is skipped, so a rerun resumes. D defaults to the current directory
# (give an absolute path otherwise), MATS to mats, MET to the bundled
# met_tables/Kazaross-XG2.xml. tools/gnubg_vs_xg.py reads the outputs.
# 100 recorded 11-point matches at 24 processes: 2-ply 2.4 min, 3-ply 14 min
# (4-ply ~10x 3-ply).
set -u
PLIES=$1; NP=$2; D=${D:-.}; MATS=${MATS:-mats}; O=$D/out_p$PLIES; mkdir -p $O
MET=${MET:-$(cd "$(dirname "$0")/.." && pwd)/met_tables/Kazaross-XG2.xml}   # met.resolve_met_path("auto")
[ -f "$MET" ] || { echo "MET not found: $MET" >&2; exit 1; }
cd $D
one() { f=$1; b=$(basename $f .mat); [ -s $O/$b.txt ] && grep -q "Overall statistics" $O/$b.txt && return
  printf "set met %s\nset threads 1\nset analysis chequerplay evaluation plies %s\nset analysis cubedecision evaluation plies %s\nimport mat %s\nanalyse match\nshow statistics match\n" "$MET" $PLIES $PLIES $f | timeout 7200 gnubg -t -q > $O/$b.txt 2>&1; }
export -f one; export O PLIES MET
echo "GNUBG_START plies=$PLIES np=$NP $(date -u +%T)"
ls $MATS/match_*.mat | xargs -P $NP -I{} bash -c "one {}"
echo "GNUBG_DONE plies=$PLIES $(date -u +%T) files=$(grep -l "Overall statistics" $O/*.txt | wc -l)"
