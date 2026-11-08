#!/bin/bash
# Undoes install-p2p-580.173.02.sh: restores the stock closed RM, UVM header and libcuda
# that it saved, replaces the P2P module options with the coexistence-only options
# (Layer A in P2P_P100_RTX50.md), and rebuilds the DKMS modules.
#
# Usage: sudo ./rollback-p2p-580.173.02.sh [backup-dir] [kernel-version]
#   backup-dir      default: the directory recorded in /var/backups/nvidia-p2p-mixed-arch.last
#   kernel-version  default: the running kernel
#
# The coexistence options stay in place because, on the proprietary driver, the RTX 5080
# needs Rm4598628 to initialize at all. Delete /etc/modprobe.d/90-nvidia-coexist.conf
# yourself only if the 5080 + P100 pair is no longer in the machine.
# Like the install script, this is a cleaned-up version of the commands used on the test
# machine. It has only been run in the same sandbox, not on a real system.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VER=580.173.02
BACKUP=${1:-$(cat /var/backups/nvidia-p2p-mixed-arch.last 2>/dev/null || true)}
KVER=${2:-$(uname -r)}
SRC=/usr/src/nvidia-$VER
LIBCUDA=/usr/lib/x86_64-linux-gnu/libcuda.so.$VER
CONF=/etc/modprobe.d/90-nvidia-p2p-mixed-arch.conf
COEXIST_CONF=/etc/modprobe.d/90-nvidia-coexist.conf

RM_STOCK=232d1c78a39c5a8791b5739dc3e252ca724e384a334bbf47f9feb1c56e21eaae
UVMH_STOCK=957c027364775f63c534dc859e6dd05ed02a4cf96247cddd18cec1fb91e0e3e2
LIBCUDA_STOCK=266916dac6c5e7e4655526570cf303a6cc017880d2552960abc66017a7c98cf4

die() { echo "error: $*" >&2; exit 1; }

check() { # file expected-sha256 label
    local actual
    actual=$(sha256sum "$1" | cut -d' ' -f1)
    [ "$actual" = "$2" ] || die "$3: $1 has SHA-256 $actual, expected $2"
}

[ "$(id -u)" -eq 0 ] || die "run as root"
[ -n "$BACKUP" ] && [ -d "$BACKUP" ] || die "backup directory not found; pass it as the first argument"

echo "== checking the saved stock files in $BACKUP"
check "$BACKUP/nv-kernel.o_binary" "$RM_STOCK" "saved closed RM"
check "$BACKUP/uvm_gpu.h" "$UVMH_STOCK" "saved UVM header"
check "$BACKUP/libcuda.so.$VER" "$LIBCUDA_STOCK" "saved libcuda"

echo "== restoring"
install -o root -g root -m 0644 "$BACKUP/nv-kernel.o_binary" "$SRC/nvidia/nv-kernel.o_binary"
install -o root -g root -m 0644 "$BACKUP/uvm_gpu.h" "$SRC/nvidia-uvm/uvm_gpu.h"
install -o root -g root -m 0644 "$BACKUP/libcuda.so.$VER" "$LIBCUDA"
ldconfig
rm -f "$CONF"
install -o root -g root -m 0644 "$HERE/modprobe/nvidia-coexist.conf" "$COEXIST_CONF"

echo "== rebuilding nvidia/$VER for $KVER"
dkms build -m nvidia -v "$VER" -k "$KVER" --force
dkms install -m nvidia -v "$VER" -k "$KVER" --force
update-initramfs -u -k "$KVER"

echo "== result"
sha256sum "$SRC/nvidia/nv-kernel.o_binary" "$SRC/nvidia-uvm/uvm_gpu.h" "$LIBCUDA" "$COEXIST_CONF"
echo "The $VER packages are still on hold; run apt-mark unhold on them before upgrading."
echo "Reboot to load the stock modules."
