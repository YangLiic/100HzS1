# B0 可选池化与两组无池化实验

现有四个配置保留，只增加一个开关：

```yaml
model:
  visual_variant: stage4  # 或 stage4_rms
  visual_pooling: false
```

- `true`（默认）：共享stage4压成2×3 → 三视角融合 → 原平均池化2×2 → Linear128→1024，最终12个视觉token。
- `false`：共享stage4压成2×3 → 三视角融合 → 直接Linear128→1024，**完全不调用池化**，最终18个视觉token。

在原`b0_stage4_fm4.yaml`或`b0_stage4_rms_fm4.yaml`中设该值，或训练时追加`model.visual_pooling=false`即可。输出网格、条件长度和KV长度自动随之更新；保存的config.json记录选择，评测读取权重配置。

本次共享stage4两组仍使用一个冻结B0和同一个stage4额外两次、一个共享可训练128→64桥接，且没有原两个ResBlock或独立MBConv。Linear均128→1024；stage4组sceneMLP1024→2048→2048、D2048；stage4_rms组视觉Pre-RMS残差MLP1024→2048→1024、D1024。DiT实现和FFN等宽约束、状态/动作/时间编码均未改。

关闭池化不增加压缩器参数；完整策略的视觉位置嵌入长度由12变18，因此需要新训练，不能把12token策略权重当18token继续训练。默认开启池化的旧四组main/EMA已逐一严格重载和FM4验证，兼容；旧产物没有修改。

共同保持beat_block_hammer/demo_clean、50demo、seed42、200epoch/8771步、三视角、minmax、BF16、标准FM/tau_token/1000时间桶，推理4step/eager/KV开启/Graph关闭/闭环RGB不翻转。每组最终main同一100固定seed闭环并保存视频，main/EMA均严格重载。只训练本次两组，旧四组不重跑。

回归61项通过，新增测试确认关闭池化时没有池化调用、自然2×3直投影及梯度正确、配置往返和旧默认值兼容；DiT、RMS模块、B0编码器、融合器和归一化实现哈希保持不变。

计划结果与上轮pool=true的stage4 32/100、stage4_rms45/100对比；实际提交身份和最终结果在独立实验笔记与结果MD中记录。
