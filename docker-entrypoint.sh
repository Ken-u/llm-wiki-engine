#!/bin/sh
set -eu

if [ -d /host-ssh ]; then
    rm -rf /root/.ssh
    mkdir -p /root/.ssh
    cp -a /host-ssh/. /root/.ssh/
    chown -R root:root /root/.ssh
    chmod 700 /root/.ssh
    find /root/.ssh -type d -exec chmod 700 {} \;
    find /root/.ssh -type f -exec chmod 600 {} \;
    if [ -f /root/.ssh/known_hosts ]; then
        chmod 644 /root/.ssh/known_hosts
    fi
fi

exec "$@"
