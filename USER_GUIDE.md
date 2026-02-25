# 黑盒免费模型代理使用说明

## 1. 这是什么
OpenRouter Free Proxy 把几十个最新的免费模型封装成一个黑盒入口：`POST /free`（或 `/free/chat/completions`、`/v1/chat/completions`）。用户只需要像对 OpenAI API 一样提问，其余模型选择、轮换、熔断、日志都由代理负责，直到所有免费模型额度全部耗尽后才会返回 503。

## 2. 快速启动
1. 克隆代码并进入 `openrouterproxy/`。
2. 创建并激活虚拟环境，安装依赖：
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. 配置 OpenRouter API Key（写入 `.env` 或导出环境变量）：
   ```bash
   export OPENROUTER_API_KEY=sk-xxxxxxxx
   ```
4. 启动服务：
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8080
   ```
   日志会显示已加载的免费模型列表以及刷新状态。

## 3. 提问方式
- 请求体与 OpenAI Chat Completions 完全一致，只需要提供 `messages` 列表：
  ```bash
  curl http://localhost:8080/free \
    -H 'Content-Type: application/json' \
    -d '{
          "messages": [
            {"role": "user", "content": "今天心情不好怎么办？"}
          ]
        }'
  ```
- 代理会自动补充当前可用模型 ID、按需要裁剪历史上下文，并把 OpenRouter 返回原封不动地转发给你。
- 如果需要连续对话，只要复用 `conversation_id` 字段或 `X-Conversation-Id` 头，代理会在本地 memory 中保留对话并根据不同模型上下文自动剪枝。

## 4. 黑盒调度策略
- 定时任务每小时刷新免费模型列表，确保池子始终跟进 OpenRouter 前台“免费”机型。
- 代理内部维护冷却/黑名单：一旦模型返回 402/429 或频繁 4xx 错误，它会进入本地黑名单并在冷却期内被跳过。
- 用户端无需知道当前用的是哪款模型，日志里会记录实际命中模型及请求/响应长度，方便排查。

## 5. 模型耗尽时会发生什么
- 当所有模型都进入冷却或黑名单时，`POST /free` 会立即返回 `503 Service Unavailable`，提示“当前没有可用模型”。
- 等待冷却结束或手动清理 `model_cache.json`（或设置 `MODEL_CACHE_RESET_ON_START=1` 后重启）即可重新开始轮训。

## 6. 常用排查
- 查看 `proxy.log` 获取每次请求/响应长度、命中模型、OpenRouter 状态码。
- `GET /healthz` 可以看到池子中每个模型的 `available`/`cooldown_until` 状态。
- 如果想完全重置状态，删除 `model_cache.json` 后重启服务。
