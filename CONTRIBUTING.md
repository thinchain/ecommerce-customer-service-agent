# 开发说明

1. 按 README 安装依赖；把 `.env.example` 复制为本地 `.env`，保留个人配置。
2. 单元测试与 Mock 集成优先使用 `python -B scripts/test_offline.py`，不启动真实模型。
3. 修改工具时同步参数校验、权限边界、角色白名单及相关测试。
4. 商品价格库存只从服务端校验字段发布；知识引用有效不等于语义验证完成。
5. 生产数据、真实订单、凭据、模型权重和原始会话不进入测试 fixtures 或提交。
6. 提交前运行 `python3 -B scripts/check_public_files.py`，并人工检查实际暂存差异。

模板应使用通用路径和占位值。个人机器笔记使用被忽略的 `README.local.md` 或 `local/`。
原始报告默认私有；公开报告按字段脱敏后放入 `reports/public/`。
发布与导出步骤见 [发布指南](docs/public-release.md)。
项目尚未选定开源许可证，公开分发前应由作者确定代码许可及第三方材料边界。
