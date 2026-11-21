# contract-agent 工作约定

> 项目级指令，只在 `contract-agent/` 下生效。优先于用户全局 `AGENTS.md`，但不覆盖 system、developer 与用户当轮的指令。

## 完成定义（可执行，不靠"看起来对了"）

- **改代码**：`uv run pytest` 必须全绿。当前沙箱环境有 8 个用例因无法创建临时目录而 ERROR，属环境限制；本机应全部通过，**新增失败一律不允许**。
- **改合同样本或任何文本导入**：先跑

  ```bash
  uv run python scripts/verify_text_source.py --source <原件> --derived <文本>
  ```

  退出码必须为 0。它只证明"派生文本忠实于源文档"，不证明源文档本身可信——来源另记 `data/samples/README.md`。
- **改文档里的数字**：核对 README / PLAN 与实测一致（测试数、语料条数、评测指标）。提交前跑 `python check-status.py`。

## 工具顺序

处理 `.doc` / `.docx` / `.pdf` 前先试 `office_convert`。

**已知例外**：政府合同模板多为 WPS 生成的复合文档，`office_convert` 会以 `invalid or truncated compound-file header` 拒绝。此时用 `scripts/verify_text_source.py` 内置的解析器，并**必须**跑上面的校验命令。

未试过 `office_convert` 之前，不得声称"没有现成工具"。

## 仓库边界

- 只写产品与设计，不写个人计划类表述
- 合同样本只用公开来源，来源、制定机关、获取日期记进 `data/samples/README.md`
- 禁止使用公司真实合同或内部资料
- 页面与文档标注"仅供参考，不构成法律意见"

## 提交习惯

- 每批一个可演示增量：代码 + 测试 + 文档同步，一起提交
- 提交信息用 `type: 英文 subject`，正文写清取舍与理由
- 设计文档放 `docs/superpowers/specs/`，实施计划放 `docs/superpowers/plans/`
- 技术债登记在 `docs/tech-debt.md`，结案时写明依据
