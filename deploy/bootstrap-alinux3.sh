#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

${SUDO} dnf -y install git wget

if ! command -v docker >/dev/null 2>&1; then
  ${SUDO} wget -O /etc/yum.repos.d/docker-ce.repo \
    http://mirrors.cloud.aliyuncs.com/docker-ce/linux/centos/docker-ce.repo
  ${SUDO} sed -i \
    's|https://mirrors.aliyun.com|http://mirrors.cloud.aliyuncs.com|g' \
    /etc/yum.repos.d/docker-ce.repo
  ${SUDO} dnf -y install dnf-plugin-releasever-adapter --repo alinux3-plus
  ${SUDO} dnf -y install \
    docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

${SUDO} systemctl enable --now docker

if ! ${SUDO} swapon --show=NAME --noheadings | grep -q .; then
  if [[ ! -f /swapfile ]]; then
    ${SUDO} fallocate -l 2G /swapfile || \
      ${SUDO} dd if=/dev/zero of=/swapfile bs=1M count=2048 status=progress
  fi
  ${SUDO} chmod 600 /swapfile
  ${SUDO} mkswap /swapfile
  ${SUDO} swapon /swapfile
  if ! grep -q '^/swapfile ' /etc/fstab; then
    echo '/swapfile none swap sw 0 0' | ${SUDO} tee -a /etc/fstab >/dev/null
  fi
fi

${SUDO} mkdir -p /srv/geochemistrypi

docker --version
docker compose version
free -h
df -h /
