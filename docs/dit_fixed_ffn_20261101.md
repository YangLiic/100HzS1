# DiT FFN固定等宽

按用户要求，DiT每层FFN始终为D→D→GELU→D，D即DiT隐藏维度。
1024维DiT使用1024→1024→GELU→1024；2048维原始模型仍使用2048→2048→GELU→2048。
tau_token和tau_AdaRMS共用这一约束，不保留独立FFN宽度参数或YAML入口。
删除ffn_hidden_size配置字段及其在DiT/Block/AdaRMSBlock中的参数和传递。
旧配置中null字段仅作无效元数据剥离；任何非空ffn_hidden_size均明确拒绝，不会静默改结构。

视觉侧不变：Linear128→1024；RMSNorm残差MLP1024→2048→GELU→1024。
本次仅修改代码/维护配置与CPU验证，不启动新训练或评测，不修改已完成实验的权重/结果/冻结快照。
16%历史组使用过宽FFN，保留原始事实；该组完整权重与当前等宽DiT尺寸不匹配，须使用原快照复现。

验证：34项CPU测试通过，覆盖两种tau注入、FM/MeanFlow兼容、前反向、KV缓存和严格重载；MeanFlow仅为CPU兼容测试，不是新增实验。
实际完整1024维模型8个DiT block均为1024→1024→GELU→1024，视觉残差MLP仍1024→2048→1024，条件token仍12个。
独立FFN参数无法从配置、DiT或block构造器开启；宽FFN张量严格加载会失败，不会被静默裁剪。
证据：runs/dit_fixed_ffn_20261101_v1/tests.log、new_architecture.json。
