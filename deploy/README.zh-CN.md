# Geochemistry Pi Online 阿里云 ECS 部署

本文档对应单台 Linux ECS、Docker Compose、Nginx 和一个 FastAPI worker 的首期部署方案。

## 1. 服务器准备

推荐 Alibaba Cloud Linux 3 或 Ubuntu 22.04，至少 4 vCPU、8 GiB 内存和 40 GiB 系统盘。安全组仅向公网开放 80/443；SSH 端口仅允许管理员固定 IP。不要对公网开放 5173 或 8000。

安装 Git、Docker Engine 和 Docker Compose v2，然后把仓库放到 `/srv/geochemistrypi`。

## 2. 创建生产环境配置

```bash
cd /srv/geochemistrypi
cp .env.production.example .env.production
git rev-parse --short=12 HEAD
openssl rand -hex 32
```

编辑 `.env.production`：

- 将 Git 提交号写入 `GEOCHEMISTRYPI_SOURCE_REVISION`；
- 为本次部署设置唯一的 `GEOCHEMISTRYPI_BUILD_ID`；
- 将随机密钥写入 `SECRET_KEY`；
- 根据 ECS 配置调整 CPU、内存和计算线程限制；
- 不要把 `.env.production` 提交到 Git。

## 3. 首次使用 HTTP 验证

正式对公网开放前，建议把安全组 80 端口暂时限制为管理员 IP。

```bash
docker compose --env-file .env.production -f docker-compose.production.yml config
docker compose --env-file .env.production -f docker-compose.production.yml build --pull
docker compose --env-file .env.production -f docker-compose.production.yml up -d
docker compose --env-file .env.production -f docker-compose.production.yml ps
curl http://127.0.0.1/healthz
curl http://127.0.0.1/api/health
```

如果当前使用的是 2 vCPU、1 GiB 内存的免费服务器，先运行：

```bash
bash deploy/bootstrap-alinux3.sh
```

并在后续 Compose 命令中加入低内存覆盖文件：

```bash
docker compose \
  --env-file .env.production \
  -f docker-compose.production.yml \
  -f docker-compose.low-memory.yml \
  build backend

docker compose \
  --env-file .env.production \
  -f docker-compose.production.yml \
  -f docker-compose.low-memory.yml \
  build web

docker compose \
  --env-file .env.production \
  -f docker-compose.production.yml \
  -f docker-compose.low-memory.yml \
  up -d
```

该覆盖文件仅用于验证部署和小数据功能。不要在 1 GiB 服务器上运行 XGBoost、多模型比较、交叉验证或较大的 Excel 数据。

检查 `/api/health` 中的 `source_revision`、`build_id` 和 `instance_id` 是否与 `.env.production` 一致。

## 4. 启用 HTTPS

准备证书和私钥，例如：

```text
/srv/geochemistrypi/certs/fullchain.pem
/srv/geochemistrypi/certs/privkey.pem
```

确认 `.env.production` 中的 `TLS_CERT_PATH` 和 `TLS_KEY_PATH` 指向上述文件，然后运行：

```bash
docker compose \
  --env-file .env.production \
  -f docker-compose.production.yml \
  -f docker-compose.https.yml \
  config

docker compose \
  --env-file .env.production \
  -f docker-compose.production.yml \
  -f docker-compose.https.yml \
  up -d
```

HTTPS 配置会把 80 自动跳转到 443。上线前检查：

```bash
curl -I http://your-domain.example
curl https://your-domain.example/api/health
```

## 5. 常用运维命令

查看状态和日志：

```bash
docker compose --env-file .env.production -f docker-compose.production.yml ps
docker compose --env-file .env.production -f docker-compose.production.yml logs --tail=200 backend
docker compose --env-file .env.production -f docker-compose.production.yml logs --tail=200 web
```

更新版本：

```bash
git fetch origin
git checkout <verified-release-tag-or-commit>
docker compose --env-file .env.production -f docker-compose.production.yml build --pull
docker compose --env-file .env.production -f docker-compose.production.yml up -d
```

停止服务但保留上传数据和结果：

```bash
docker compose --env-file .env.production -f docker-compose.production.yml down
```

不要在生产环境执行 `docker compose down -v`，该命令会删除 `geochemistrypi-runtime` 数据卷。

## 6. 当前部署边界

- 后端固定为一个 worker，因为任务状态和队列目前保存在进程内存中。
- Nginx 暂时允许长连接以兼容现有同步计算请求。
- 正式开放前仍需完成队列长度上限、任务文件自动清理和异步任务提交改造。
