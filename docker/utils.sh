
# --- Microduck Workshop shell helpers (appended to /root/.bashrc) ---
source /root/env 2>/dev/null
cd /workspace/Microduck-Workshop 2>/dev/null
export PS1='(microduck) \w\$ '

# Persist a setting in the mounted docker/env file, like the SO-101 workshop.
setenv() {
  local key val
  if [ -z "$1" ]; then
    echo "Usage: setenv KEY VALUE | setenv KEY=VALUE | setenv -d KEY"
    cat /root/env 2>/dev/null
    return 1
  fi
  if [ "$1" = "-d" ]; then
    sed "/^export ${2}=/d" /root/env > /tmp/env.tmp && cat /tmp/env.tmp > /root/env
    unset "$2"
    return 0
  fi
  if [ -n "$2" ]; then key="$1"; val="$2"; else key="${1%%=*}"; val="${1#*=}"; fi
  if grep -q "^export ${key}=" /root/env 2>/dev/null; then
    sed "s|^export ${key}=.*|export ${key}=${val}|" /root/env > /tmp/env.tmp && cat /tmp/env.tmp > /root/env
  else
    printf '\n%s\n' "export ${key}=${val}" >> /root/env
  fi
  source /root/env
  echo "${key}=${val}"
}
