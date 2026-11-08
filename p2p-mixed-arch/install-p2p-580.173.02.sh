#!/bin/bash
# Applies the RTX 5080 + Tesla P100 P2P patch set (Layer B in P2P_P100_RTX50.md) to an
# installed Ubuntu NVIDIA 580.173.02 driver (packages 580.173.02-0ubuntu0.26.04.1) and
# rebuilds the DKMS modules. It patches files from your own driver install; it ships no
# NVIDIA binaries. Every input and output is checked against a known SHA-256, so any
# other driver build is refused before a file is changed.
#
# Usage: sudo ./install-p2p-580.173.02.sh [kernel-version]   (default: the running kernel)
# Needs: python3, python3-pyelftools, patch, dkms, kernel headers. Reboot afterwards.
#
# This is a cleaned-up version of the commands used on the test machine (August and
# November 2026). The script itself has only been run in a sandbox, with the system paths
# redirected into a temporary directory and dkms, update-initramfs, ldconfig, apt-mark
# and modinfo stubbed; it has not been run on a real system. See the "Install" section
# of P2P_P100_RTX50.md for the step-by-step equivalent.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VER=580.173.02
KVER=${1:-$(uname -r)}
SRC=/usr/src/nvidia-$VER
RM=$SRC/nvidia/nv-kernel.o_binary
UVMH=$SRC/nvidia-uvm/uvm_gpu.h
LIBCUDA=/usr/lib/x86_64-linux-gnu/libcuda.so.$VER
CONF=/etc/modprobe.d/90-nvidia-p2p-mixed-arch.conf
COEXIST_CONF=/etc/modprobe.d/90-nvidia-coexist.conf
BACKUP=/var/backups/nvidia-p2p-mixed-arch-$(date +%Y%m%d-%H%M%S)

# Stock inputs (Ubuntu 580.173.02-0ubuntu0.26.04.1) and the patched outputs that were
# installed and tested on the test machine.
RM_STOCK=232d1c78a39c5a8791b5739dc3e252ca724e384a334bbf47f9feb1c56e21eaae
RM_P2P=3bb9480f71a59aad3e8253fedae251480a4d65606429b584543054e6efd970e1
UVMH_STOCK=957c027364775f63c534dc859e6dd05ed02a4cf96247cddd18cec1fb91e0e3e2
UVMH_P2P=33bf8f2e40c307ce0f7c379ea209e437f50a7ae069786e2c90a85163ef57c35d
LIBCUDA_STOCK=266916dac6c5e7e4655526570cf303a6cc017880d2552960abc66017a7c98cf4
LIBCUDA_P2P=9e1ffa0f9eec3b6eac17ebaddc88f2f27a3bc1de498cfd8ddd01dc154a27ed91

die() { echo "error: $*" >&2; exit 1; }

check() { # file expected-sha256 label
    local actual
    actual=$(sha256sum "$1" | cut -d' ' -f1)
    [ "$actual" = "$2" ] || die "$3: $1 has SHA-256 $actual, expected $2"
}

[ "$(id -u)" -eq 0 ] || die "run as root"
for tool in python3 patch dkms sha256sum update-initramfs; do
    command -v "$tool" >/dev/null || die "$tool is required"
done
python3 -c 'import elftools' 2>/dev/null || die "pyelftools is required (apt install python3-pyelftools)"
[ -d "/lib/modules/$KVER/build" ] || die "kernel headers for $KVER are missing"
for f in "$RM" "$UVMH" "$LIBCUDA"; do
    [ -f "$f" ] || die "$f not found; install the $VER driver packages first"
done

echo "== checking the stock $VER files"
check "$RM" "$RM_STOCK" "closed RM (not the stock $VER file, or already patched)"
check "$UVMH" "$UVMH_STOCK" "UVM header (not the stock $VER file, or already patched)"
check "$LIBCUDA" "$LIBCUDA_STOCK" "libcuda (not the stock $VER file, or already patched)"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "== building the patched files in $WORK"
python3 "$HERE/patch_closed_rm_p2p.py" "$RM" "$WORK/nv-kernel.o_binary"
check "$WORK/nv-kernel.o_binary" "$RM_P2P" "patched closed RM"
python3 "$HERE/patch_libcuda_peer_arch.py" --source "$LIBCUDA" --output "$WORK/libcuda.so.$VER"
check "$WORK/libcuda.so.$VER" "$LIBCUDA_P2P" "patched libcuda"
mkdir -p "$WORK/uvm/nvidia-uvm"
cp "$UVMH" "$WORK/uvm/nvidia-uvm/uvm_gpu.h"
patch -p1 -d "$WORK/uvm" < "$HERE/uvm_gpu.h-coherence.patch"
check "$WORK/uvm/nvidia-uvm/uvm_gpu.h" "$UVMH_P2P" "patched UVM header"

echo "== saving the stock files to $BACKUP"
mkdir -p "$BACKUP"
cp -a "$RM" "$UVMH" "$LIBCUDA" "$BACKUP/"
for f in "$CONF" "$COEXIST_CONF"; do
    if [ -e "$f" ]; then
        cp -a "$f" "$BACKUP/"
    fi
done
echo "$BACKUP" > /var/backups/nvidia-p2p-mixed-arch.last

echo "== installing"
install -o root -g root -m 0644 "$WORK/nv-kernel.o_binary" "$RM"
install -o root -g root -m 0644 "$WORK/uvm/nvidia-uvm/uvm_gpu.h" "$UVMH"
install -o root -g root -m 0644 "$WORK/libcuda.so.$VER" "$LIBCUDA"
ldconfig
# The P2P file carries the coexistence key too; a second NVreg_RegistryDwords line
# from the Layer A file would compete with it.
rm -f "$COEXIST_CONF"
install -o root -g root -m 0644 "$HERE/modprobe/nvidia-p2p-mixed-arch.conf" "$CONF"

echo "== rebuilding nvidia/$VER for $KVER"
dkms build -m nvidia -v "$VER" -k "$KVER" --force
dkms install -m nvidia -v "$VER" -k "$KVER" --force
update-initramfs -u -k "$KVER"

echo "== holding the $VER packages (a package update would overwrite the patched files)"
apt-mark hold $(dpkg-query -W -f='${Package} ${Version} ${Status}\n' |
    awk -v v="$VER" 'index($2, v) == 1 && $NF == "installed" {print $1}')

echo "== result"
sha256sum "$RM" "$UVMH" "$LIBCUDA" "$CONF"
modinfo -k "$KVER" -F version nvidia
echo "Stock files saved in $BACKUP (rollback: rollback-p2p-580.173.02.sh)."
echo "Reboot, then run the checks in the Verify section of P2P_P100_RTX50.md."
