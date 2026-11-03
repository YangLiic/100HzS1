# B0共享stage4：融合前可选2×2池化

保留原B0预训练冻结、同一个stage4额外调用两次和共享可训练128→64入口桥接。
新配置在两次额外stage4之后、三视角融合之前增加AdaptiveAvgPool2d((2,2))：
128×8×10→128×4×5→128×2×3→128×2×2，再以每视角4token/三视角12token进入融合。
池化没有可训练参数或state_dict张量；Linear128→1024、sceneMLP1024→2048→GELU→2048及DiT2048保持原共享stage4配方。
标准FM/tau_token/1000桶/4steps、原数据/训练设置不变。只准备代码/YAML，尚未启动新训练或评测。

原2×3/6token配置继续可用；stage4_pool_shape默认null，不静默改变旧模型。
从18变12个条件token会改变DiT条件位置嵌入长度，所以新2×2完整策略不可直接strict加载旧6token完整策略。
视觉编码器参数形状相同，可严格复用视觉权重；完整新实验应从独立配置训练，不宣称旧成功率保持。

融合后共享stage4分支、专属YAML和实验说明已退役。用户授权删除其权重、临时权重、RoboTwin结果/视频和专属run。
当前RMSNorm残差MLP实验是另一组：stage4_repeats=0，使用独立冻结快照/配置/权重；不依赖被删除组。
保留运行中的快照，兼容性以当前开发代码CPU严格重载其main/EMA和回归测试验证。

验证：33项CPU测试通过，覆盖池化位置/精确数值/0参数、桥接梯度、原6token行为、FM4采样、KV等价和严格重载。
改后的开发代码已CPU严格重载实际RMSNorm实验最终main/EMA；其4次FM前向和冻结B0核验通过。
清理删除24份权重文件及100个视频，共17,925,383,616字节；保留组权重与977个运行快照源码哈希未变。
清理凭据：runs/cleanup_postfusion_stage4_20261101_v1/{manifest,deletion_verified,rms_current_code_strict_reload}.json。
