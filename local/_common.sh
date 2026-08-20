# 所有 local/*.sh 的公共头部。用法：source "$(dirname "$0")/_common.sh"
#
# 做三件事：
#   1. 自动推导 REPO_DIR（= local/ 的父目录），换机器 / 改路径都不用动脚本
#   2. 读入 local/env.sh 里的本机配置
#   3. cd 到 REPO_DIR —— Hydra 的 conf/ 查找和 python3 -m data.xxx 的模块导入
#      都要求工作目录是仓库根，否则从 local/ 里直接跑会报错

set -euo pipefail

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${LOCAL_DIR}")"
export REPO_DIR

if [ ! -f "${LOCAL_DIR}/env.sh" ]; then
    echo "ERROR: 缺少 ${LOCAL_DIR}/env.sh" >&2
    echo "       先执行: cp local/env.sh.example local/env.sh 然后按本机情况修改" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "${LOCAL_DIR}/env.sh"

cd "${REPO_DIR}"
